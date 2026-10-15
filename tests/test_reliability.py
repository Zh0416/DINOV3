import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pvseg.attention import save_attention_cache
from pvseg.metrics import Confusion
from pvseg.models.adapter import PVAwareAdapter, LegacyPVAwareAdapter
from pvseg.reliability import TeacherReliability, filter_attention


class ReliabilityTests(unittest.TestCase):
    def test_empty_gate_identity_even_after_learning_biases(self):
        adapter = PVAwareAdapter(8, 4)
        with torch.no_grad():
            adapter.mlp[-1].weight.fill_(.2)
            adapter.mlp[-1].bias.fill_(1.)
        feature = torch.randn(2, 8, 4, 4)
        self.assertTrue(torch.equal(adapter(feature, torch.zeros(2, 4, 4)), feature))
        gate = torch.ones(2, 4, 4)
        gate[:, :, :2] = 0
        updated = adapter(feature, gate)
        self.assertTrue(torch.equal(updated[:, :, :, :2], feature[:, :, :, :2]))
        self.assertFalse(torch.equal(updated, feature))

    def test_legacy_empty_attention_is_not_identity(self):
        feature = torch.randn(1, 8, 4, 4)
        self.assertFalse(torch.equal(LegacyPVAwareAdapter(8, 4)(feature, torch.zeros(1, 4, 4)), feature))

    def test_calibration_ignores_validation_and_test_labels(self):
        records, rows = [], []
        for i in range(15):
            records.append({"sample_id": str(i), "pv_exist": True, "status": "ok", "confidence": .95,
                            "bbox": [[i*5, 0, 200, 200]]})
            rows.append({"sample_id": str(i), "split": "train" if i < 10 else "test", "useful_localization": i % 2 == 0})
        original = TeacherReliability().fit(records, rows)
        mutated = copy.deepcopy(rows)
        for row in mutated[10:]:
            row["useful_localization"] = not row["useful_localization"]
        changed = TeacherReliability().fit(records, mutated)
        np.testing.assert_array_equal(original.weights, changed.weights)
        self.assertTrue(all(int(x) < 10 for x in original.training_ids))

    def test_untrusted_records_and_disagreement_get_zero_attention(self):
        record = {"sample_id": "x", "status": "ok", "pv_exist": True, "confidence": .4, "bbox": [[1, 1, 20, 20]]}
        with tempfile.TemporaryDirectory() as directory:
            save_attention_cache([record], Path(directory) / "cache", 32, 4, 2, confidence_threshold=.5)
            self.assertFalse(np.load(Path(directory) / "cache/x.npy").any())
        self.assertFalse(filter_attention(np.ones((4, 4)), np.zeros((4, 4)), .9, .95).any())

    def test_conditional_miss_and_false_alarm_denominators(self):
        meter = Confusion()
        target = torch.tensor([[1, 0], [0, 0]])
        meter.update(torch.tensor([[0, 0], [0, 1]]), target)
        meter.update(torch.ones(2, 2), torch.zeros(2, 2))
        result = meter.compute()
        self.assertEqual(result["overlap_miss_rate"], 1.)
        self.assertEqual(result["image_false_alarm_rate"], 1.)
        self.assertEqual(result["positive_images"], 1)


if __name__ == "__main__":
    unittest.main()
