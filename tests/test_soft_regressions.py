import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pvseg.config import load_config
from pvseg.final_evaluation import evaluate_sealed
from pvseg.label_free_data import RGBSample, PseudoFeatureDataset
from pvseg.label_free_training import SOFT_VARIANTS, _optimizer_step, save_predictions
from pvseg.models import CachedFeatureSegmenter
from pvseg.soft_semantic import build_soft_pseudo_targets, fuse_soft_attention
from pvseg.teacher import parse_teacher_response
from pvseg.utils import atomic_write_json


def cell(probability=.95, status="positive", row=1, col=1):
    return {"row": row, "col": col, "pv_probability": probability, "status": status}


def view(*cells, **geometry):
    return {"status": "ok", "grid": list(cells), **geometry}


class SoftRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def build(self, record, size=12, **overrides):
        settings = {"feature_size": size, "gaussian_sigma_patches": 0., "boundary_pixels": 1,
                    "min_pseudo_retention": .5, "positive_weight_floor": .35, **overrides}
        features = np.zeros((2, size, size), np.float32)
        features[1] = 1
        with patch("pvseg.soft_semantic.rgb_seeds", return_value=(np.ones((size, size), bool), np.zeros((size, size), bool))):
            return build_soft_pseudo_targets(features, "unused", np.eye(2), record, settings)

    def test_compact_grid_has_all_positions_and_statuses(self):
        rows = [[[.9, "p"], [.95, "n"], [.8, "u"], [.7, "p"]] for _ in range(4)]
        result = parse_teacher_response(json.dumps({"pv_exist": True, "grid": rows}))
        self.assertTrue(result["grid_complete"])
        self.assertEqual(result["grid_cell_count"], 16)
        self.assertEqual(result["grid"][-1]["row"], 3)
        self.assertEqual(result["grid"][1]["status"], "negative")
        self.assertEqual(result["grid"][2]["status"], "unknown")
        self.assertNotIn("parse_warning", result)

    def test_compact_truncation_keeps_only_finished_cells(self):
        raw = '{"pv_exist":true,"grid":[[[0.9,"p"],[0.1,"u"],[0,"n"],[0.8,"p"]],[[0.7,"p"],[0.6,"'
        result = parse_teacher_response(raw)
        self.assertEqual(result["grid_cell_count"], 5)
        self.assertEqual((result["grid"][-1]["row"], result["grid"][-1]["col"]), (1, 0))
        self.assertFalse(result["grid_complete"])

    def test_flat_compact_cloud_negative_response_keeps_all_cells_negative(self):
        raw = '''{
          "pv_exist": false,
          "image_confidence": 0.9,
          "grid": [
            [0.95,"n"], [0.95,"n"], [0.95,"n"], [0.95,"n"],
            [0.95,"n"], [0.95,"n"], [0.95,"n"], [0.95,"n"],
            [0.95,"n"], [0.95,"n"], [0.95,"n"], [0.95,"n"],
            [0.95,"n"], [0.95,"n"], [0.95,"n"], [0.95,"n"]
          ]
        }'''
        result = parse_teacher_response(raw)
        self.assertTrue(result["grid_complete"])
        self.assertEqual(result["grid_cell_count"], 16)
        self.assertFalse(result["pv_exist"])
        self.assertTrue(all(item["status"] == "negative" for item in result["grid"]))
        self.assertNotIn("parse_warning", result)
        record = {"views": {"original": {**result, "status": "ok"}}}
        attention, _ = fuse_soft_attention(record, {"feature_size": 8, "gaussian_sigma_patches": 0.})
        np.testing.assert_allclose(attention, .2)

    def test_flat_compact_grid_matches_nested_row_major_order(self):
        pairs = [[index / 16, ("p", "n", "u")[index % 3]] for index in range(16)]
        flat = parse_teacher_response(json.dumps({"grid": pairs}))
        nested = parse_teacher_response(json.dumps({"grid": [pairs[i:i+4] for i in range(0, 16, 4)]}))
        self.assertEqual(flat["grid"], nested["grid"])
        self.assertTrue(flat["grid_complete"])

    def test_partial_and_oversized_flat_compact_grids_are_not_complete(self):
        for count, expected_cells in ((15, 15), (17, 0)):
            with self.subTest(count=count):
                result = parse_teacher_response(json.dumps({"grid": [[.9, "p"]] * count}))
                self.assertFalse(result["grid_complete"])
                self.assertEqual(result["grid_cell_count"], expected_cells)

    def test_flat_compact_truncation_keeps_only_finished_prefix(self):
        raw = '{"grid":[[0.9,"p"],[0.1,"u"],[0,"n"],[0.8,"p"],[0.7,"p"],[0.6,"'
        result = parse_teacher_response(raw)
        self.assertEqual(result["grid_cell_count"], 5)
        self.assertEqual((result["grid"][-1]["row"], result["grid"][-1]["col"]), (1, 0))
        self.assertEqual(result["parse_warning"], "truncated_json_recovered_grid")
        self.assertFalse(result["grid_complete"])
        malformed = parse_teacher_response('{"grid":[[0.9,"p"],BROKEN,[0.8,"p"]')
        self.assertEqual(malformed["grid_cell_count"], 1)

    def test_explicit_positive_negative_conflict_is_ignored(self):
        record = {"views": {"positive": view(cell()), **{f"negative{i}": view(cell(.95, "negative")) for i in range(5)}}}
        arrays, diagnostic = self.build(record)
        self.assertEqual(diagnostic["teacher_conflict_pixels"], 9)
        self.assertEqual(diagnostic["positive_pixels"], 0)
        self.assertTrue((arrays["target_soft"][3:6, 3:6] == 255).all())
        self.assertTrue((arrays["weight_soft"][3:6, 3:6] == 0).all())
        self.assertAlmostEqual(float(arrays["attention_soft"].max()), .2, places=5)

    def test_malformed_compact_cell_does_not_shift_later_coordinates(self):
        result = parse_teacher_response('{"grid":[[[0.9,"p"],BROKEN,[0.8,"p"]')
        self.assertEqual(result["grid_cell_count"], 1)

    def test_unrelated_crop_does_not_vote_against_positive(self):
        negatives = [cell(0., "negative", r, c) for r in range(4) for c in range(4)]
        record = {"views": {"original": view(cell(row=3, col=3)),
                            "local": view(*negatives, crop_box=[0, 0, 512, 512])}}
        attention, diagnostic = fuse_soft_attention(record, {"feature_size": 8, "gaussian_sigma_patches": 0.})
        self.assertEqual(diagnostic["teacher_conflict_pixels"], 0)
        self.assertAlmostEqual(float(attention[-1, -1]), .95, places=5)

    def test_local_consistency_uses_observed_votes(self):
        record = {"views": {"strong": view(cell()), "weak1": view(cell(.2)), "weak2": view(cell(.2)),
                            "unknown": view(cell(.99, "unknown"))}}
        strict, diag = self.build(record, local_consistency=.45)
        permissive, _ = self.build(record, local_consistency=.3)
        self.assertEqual(diag["teacher_inconsistent_pixels"], 9)
        self.assertFalse((strict["target_soft"] == 1).any())
        self.assertEqual(int((permissive["target_soft"] == 1).sum()), 9)

    def test_retention_fallback_survives_boundary_mask(self):
        arrays, diagnostic = self.build({"views": {"original": view(cell())}})
        self.assertEqual(diagnostic["raw_candidate_pixels"], 9)
        self.assertAlmostEqual(diagnostic["pseudo_retention_before_fallback"], 1/9)
        self.assertTrue(diagnostic["retention_fallback"])
        self.assertEqual(diagnostic["pseudo_retention"], 1.)
        positive = arrays["target_soft"] == 1
        np.testing.assert_array_equal(arrays["teacher_core"], positive)
        restored = positive & ~arrays["teacher_eroded_core"].astype(bool)
        self.assertTrue(np.allclose(arrays["weight_soft"][restored], .35))
        self.assertTrue((arrays["weight_soft"][arrays["target_soft"] == 255] == 0).all())

    def test_single_pixel_candidate_is_not_erased(self):
        arrays, diagnostic = self.build({"views": {"original": view(cell())}}, size=4)
        self.assertTrue(diagnostic["retention_fallback"])
        self.assertEqual(diagnostic["positive_pixels"], 1)
        self.assertEqual(diagnostic["pseudo_retention"], 1.)
        self.assertEqual(arrays["target_soft"][1, 1], 1)

    def test_wide_outer_boundary_stays_ignored_after_fallback(self):
        from scipy.ndimage import binary_dilation
        size = 20
        features = np.zeros((2, size, size), np.float32)
        features[0] = 1
        with patch("pvseg.soft_semantic.rgb_seeds", return_value=(np.zeros((size, size), bool), np.ones((size, size), bool))):
            arrays, diagnostic = build_soft_pseudo_targets(features, "unused", np.eye(2),
                {"views": {"original": view(cell())}},
                {"feature_size": size, "boundary_pixels": 2, "gaussian_sigma_patches": 0.})
        self.assertTrue(diagnostic["retention_fallback"])
        candidate = arrays["teacher_candidate"].astype(bool)
        outer = binary_dilation(candidate, iterations=2) & ~candidate
        self.assertTrue((arrays["target_soft"][outer] == 255).all())
        self.assertTrue((arrays["weight_soft"][outer] == 0).all())

    def test_amp_overflow_reduces_scale_without_counting_an_update(self):
        class Overflow(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.sum()
            @staticmethod
            def backward(ctx, gradient):
                return gradient * float("inf")
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.)
        before = model.weight.detach().clone()
        # Keep the loss finite while simulating a scaled-backward overflow.
        loss = Overflow.apply(model.weight.reshape(()))
        self.assertFalse(_optimizer_step(loss, model, optimizer, scaler))
        self.assertEqual(scaler.get_scale(), 4.)
        self.assertFalse(optimizer.state)
        torch.testing.assert_close(model.weight, before)
        optimizer.zero_grad()
        self.assertTrue(_optimizer_step(model.weight.square().sum(), model, optimizer, scaler))
        self.assertEqual(int(optimizer.state[model.weight]["step"]), 1)

    def test_adapter_internal_parameters_receive_gradient_after_first_step(self):
        torch.manual_seed(12)
        model = CachedFeatureSegmenter(8, 16, 8, use_adapter=True, adapter_mode="soft_film")
        adapter = model.adapter
        optimizer = torch.optim.SGD(adapter.parameters(), lr=.1)
        features, attention = torch.randn(2, 8, 4, 4), torch.rand(2, 4, 4)
        target = torch.randn_like(features)
        branch = adapter.attention_encoder[0].weight
        ((adapter(features, attention) - target)**2).mean().backward()
        self.assertGreater(float(adapter.gamma.grad.abs()), 0.)
        self.assertEqual(float(branch.grad.abs().sum()), 0.)
        optimizer.step()
        optimizer.zero_grad()
        ((adapter(features, attention) - target)**2).mean().backward()
        self.assertGreater(float(branch.grad.abs().sum()), 0.)

    def test_capacity_control_uses_adapter_and_constant_attention(self):
        self.assertEqual(SOFT_VARIANTS["semantic_capacity"], (True, "soft", "base"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "feature_cache").mkdir()
            (root / "pseudo_cache").mkdir()
            torch.save(torch.randn(8, 4, 4), root / "feature_cache/s.pt")
            np.savez(root / "pseudo_cache/s.npz", target_soft=np.ones((4, 4), np.uint8),
                     attention_soft=np.arange(16).reshape(4, 4)/16)
            sample = RGBSample("s", "unused", "s", "train")
            dataset = PseudoFeatureDataset([sample], root, "soft", "base", base_attention=.27)
            torch.testing.assert_close(dataset[0]["attention"], torch.full((4, 4), .27))

    def test_bad_report_fails_before_annotation_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_json(root / "rgb_manifest.json", {"samples": [RGBSample("s", "unused", "s", "test").to_dict()]})
            atomic_write_json(root / "reports/pseudo_reliability.json", {"per_image": [{"sample_id": "s", "accepted": True}]})
            with patch("pvseg.final_evaluation.verify_seal", return_value={"runs": []}), patch("pvseg.final_evaluation.discover_samples") as discover, patch("pvseg.final_evaluation.Image.open") as opened:
                with self.assertRaisesRegex(ValueError, "Invalid pseudo report"):
                    evaluate_sealed({"protocol": {"mode": "soft_semantic"}}, root, logging.getLogger(__name__))
            discover.assert_not_called()
            opened.assert_not_called()

    def test_nonfinite_prediction_is_not_saved(self):
        class InvalidModel(torch.nn.Module):
            def forward(self, *args):
                return torch.full((1, 2, 4, 4), float("nan"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = [{"sample_id": "s", "feature": torch.zeros(8, 4, 4), "attention": torch.zeros(4, 4)}]
            with self.assertRaisesRegex(RuntimeError, "Non-finite prediction"):
                save_predictions(InvalidModel(), dataset, torch.device("cpu"), root, 16, .5)
            self.assertFalse(list(root.glob("*.png")))

    def test_modelscope_config_inherits_algorithm_without_kaggle_paths(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/modelscope_soft_semantic.yaml")
        self.assertTrue(config["teacher"]["require_complete_grid"])
        self.assertEqual(config["teacher"]["max_new_tokens"], 512)
        self.assertIn("semantic_capacity", config["ablation"]["variants"])
        self.assertNotIn("/kaggle", json.dumps(config))
        self.assertTrue(Path(config["data"]["exclude_rgb_hash_file"]).is_file())
        self.assertTrue(Path(config["data"]["exclude_manifest_file"]).is_absolute())

    def test_config_inheritance_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cycle.yaml"
            path.write_text("extends: cycle.yaml", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inheritance cycle"):
                load_config(path)
