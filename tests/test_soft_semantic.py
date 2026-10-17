import unittest
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from pvseg.losses import IgnoreAwareDiceFocalLoss
from pvseg.final_evaluation import evaluate_sealed
from pvseg.label_free_data import load_rgb_manifest, manual_label_firewall, prepare_rgb_manifest
from pvseg.label_free_training import train_label_free
from pvseg.models import CachedFeatureSegmenter
from pvseg.multiview_teacher import generate_grid_teacher
from pvseg.pseudo_labels import generate_pseudo_cache
from pvseg.prediction_seal import seal_predictions
from pvseg.soft_semantic import build_soft_pseudo_targets, fuse_soft_attention
from pvseg.teacher import parse_teacher_response
from pvseg.utils import atomic_write_json, read_json


class SoftSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_grid_parser_preserves_unknown(self):
        payload = '{"pv_exist":true,"image_confidence":0.8,"grid":[[{"pv_probability":0.9,"status":"positive"},{"pv_probability":0.1,"status":"unknown"}]]}'
        parsed = parse_teacher_response(payload)
        self.assertEqual(len(parsed["grid"]), 2)
        self.assertEqual(parsed["grid"][1]["status"], "unknown")

    def test_soft_attention_has_nonzero_base(self):
        record = {"views": {"original": {
            "status": "ok", "grid": [{"row": 1, "col": 1, "pv_probability": .9, "status": "positive"}]
        }}}
        attention, diagnostic = fuse_soft_attention(record, {
            "feature_size": 8, "grid_size": 4, "base_attention": .2, "gaussian_sigma_patches": .5
        })
        self.assertEqual(attention.shape, (8, 8))
        self.assertGreaterEqual(float(attention.min()), .2)
        self.assertGreater(float(attention.max()), .2)
        self.assertEqual(diagnostic["teacher_views_used"], 1)

    def test_recovered_grid_warning_is_still_soft_evidence(self):
        record = {"views": {"original": {
            "status": "ok", "parse_warning": "truncated_json_recovered_grid",
            "grid": [{"row": 1, "col": 1, "pv_probability": .9, "status": "positive"}],
        }}}
        attention, diagnostic = fuse_soft_attention(record, {
            "feature_size": 8, "grid_size": 4, "base_attention": .2, "gaussian_sigma_patches": .5
        })
        self.assertGreater(float(attention.max()), .2)
        self.assertEqual(diagnostic["teacher_views_used"], 1)

    def test_unknown_and_negative_grids_do_not_become_positive_after_normalization(self):
        record = {"views": {
            "unknown": {"status": "ok", "grid": [
                {"row": 0, "col": 0, "pv_probability": .1, "status": "unknown"},
            ]},
            "negative": {"status": "ok", "grid": [
                {"row": 1, "col": 1, "pv_probability": .9, "status": "negative"},
            ]},
        }}
        attention, _ = fuse_soft_attention(record, {
            "feature_size": 8, "grid_size": 4, "base_attention": .2, "gaussian_sigma_patches": .5
        })
        self.assertAlmostEqual(float(attention.max()), .2, places=5)

    def test_high_probability_unknown_cannot_seed_positive_targets(self):
        size = 64
        record = {"views": {"original": {
            "status": "ok", "grid": [{"row": 1, "col": 1, "pv_probability": .95, "status": "unknown"}]
        }}}
        feature = np.zeros((2, size, size), np.float32)
        feature[1] = 1.  # Even independent appearance support cannot turn abstention into semantic evidence.
        settings = {"feature_size": size, "grid_size": 4, "base_attention": .2,
                    "positive_probability": .55, "feature_margin": .04, "gaussian_sigma_patches": 2.}
        with patch("pvseg.soft_semantic.rgb_seeds", return_value=(np.ones((size, size), bool), np.zeros((size, size), bool))):
            arrays, diagnostic = build_soft_pseudo_targets(feature, "unused", np.eye(2), record, settings)
        self.assertTrue(diagnostic["teacher_abstained"])
        self.assertEqual(diagnostic["positive_pixels"], 0)
        self.assertTrue((arrays["target_soft"] == 255).all())

    def test_positive_view_still_supplies_evidence_where_another_view_is_unknown(self):
        record = {"views": {
            "original": {"status": "ok", "grid": [{"row": 1, "col": 1, "pv_probability": .95, "status": "unknown"}]},
            "local": {"status": "ok", "grid": [{"row": 1, "col": 1, "pv_probability": .8, "status": "positive"}]},
        }}
        attention, diagnostic = fuse_soft_attention(record, {"feature_size": 64, "gaussian_sigma_patches": 2.})
        self.assertEqual(diagnostic["teacher_views_used"], 1)
        self.assertGreater(float(attention.max()), .7)

    def test_zero_initialized_adapter_is_identity(self):
        model = CachedFeatureSegmenter(16, 16, 8, use_adapter=True, adapter_mode="soft_film")
        features = torch.randn(2, 16, 8, 8)
        attention = torch.full((2, 8, 8), .2)
        adapted = model.adapter(features, attention)
        torch.testing.assert_close(adapted, features)
        self.assertEqual(float(model.adapter.gamma.detach()), 0.0)

    def test_focal_ignores_unknown_pixels(self):
        logits = torch.randn(1, 2, 3, 3, requires_grad=True)
        target = torch.tensor([[[1, 0, 255], [255, 255, 255], [0, 1, 0]]])
        weight = torch.ones(1, 3, 3)
        loss, parts = IgnoreAwareDiceFocalLoss()(logits, target, weight)
        loss.backward()
        self.assertTrue(np.isfinite(float(loss.detach())))
        self.assertEqual(float(logits.grad[:, :, 0, 2].abs().sum()), 0.0)
        self.assertEqual(parts["valid_pixels"], 5)

    def test_soft_protocol_trains_from_machine_targets_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            categories = ["正样本", "困难正样本", "负样本"]
            for category_index, category in enumerate(categories):
                image_dir = root / "pv-gf2" / category / "images"
                label_dir = image_dir.parent / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir()
                for index in range(2):
                    rgb = np.full((16, 16, 3), 230, np.uint8)
                    rgb[:, :8] = [35, 40, 65]
                    rgb[0, 0] = [30 + index, 40 + category_index, 65]
                    Image.fromarray(rgb).save(image_dir / f"{index}.bmp")
                    Image.fromarray((rgb[:, :, 0] < 100).astype(np.uint8) * 255).save(label_dir / f"{index}.bmp")
            config = {
                "seed": 9, "protocol": {"name": "synthetic_soft", "mode": "soft_semantic", "label_usage": "final_evaluation_only"},
                "data": {"input_root": str(root), "dataset_slug": "pv-gf2", "categories": categories,
                         "image_dir_name": "images", "label_dir_names": ["labels"], "extensions": [".bmp"],
                         "image_size": 16, "max_samples": 6, "train_fraction": 2/3,
                         "rgb_staging_dir": str(root / "staged")},
                "dino": {"expected_hidden_dim": 8},
                "teacher": {"model_id": "synthetic", "prompt": "compact grid", "local_views": False,
                            "require_complete_grid": True},
                "reliability": {"feature_size": 4, "grid_size": 4, "positive_probability": .55,
                                "negative_probability": .18, "feature_margin": 0., "min_seed_pixels": 1,
                                "boundary_pixels": 0, "gaussian_sigma_patches": .5, "base_attention": .2,
                                "min_pseudo_retention": .5, "positive_weight_floor": .35, "unknown_weight": .15},
                "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "decoder_channels": 16,
                             "adapter_bottleneck": 8, "learning_rate": .001, "weight_decay": .0001,
                             "dice_weight": 1., "focal_weight": 1., "focal_gamma": 2., "focal_alpha": .25,
                             "consistency_weight": 0., "adapter_mode": "soft_film", "threshold": .5,
                             "visualization_samples": 1, "decode_at_image_size": True},
                "ablation": {"seeds": [42], "variants": ["baseline_decoder", "semantic_adapter", "semantic_capacity", "shuffled_semantic"]},
                "runtime": {"output_dir": str(output), "resume": True},
            }
            logger = logging.getLogger("soft_protocol")
            with manual_label_firewall():
                prepare_rgb_manifest(config, output, logger)
                samples = load_rgb_manifest(output / "rgb_manifest.json")
                (output / "feature_cache").mkdir(parents=True)
                for sample in samples:
                    feature = torch.zeros(8, 4, 4)
                    feature[0] = 1.
                    torch.save(feature, output / "feature_cache" / f"{sample.sample_id}.pt")
                records = []
                for sample in samples:
                    views = {}
                    for name in ("original", "hflip"):
                        grid = [[[.1, "u"] for _ in range(4)] for _ in range(4)]
                        grid[1][2 if name == "hflip" else 1] = [.95, "p"]
                        raw = json.dumps({"pv_exist": True, "grid": grid})
                        views[name] = {**parse_teacher_response(raw), "status": "ok", "flipped": name == "hflip"}
                    records.append({"sample_id": sample.sample_id, "views": views})
                atomic_write_json(output / "semantic_prompt.json", {"teacher_model": "synthetic", "records": records})
                teacher_report = generate_grid_teacher(config, output, logger)
                self.assertEqual(teacher_report["new_inferences"], 0)
                self.assertEqual(teacher_report["complete_grid_views"], 12)
                generate_pseudo_cache(config, output, logger)
                result = train_label_free(config, output, logger, device="cpu")
                self.assertEqual(len(result["runs"]), 4)
                self.assertTrue(all(sum(e["optimizer_steps"] for e in run["history"]) >= 2 for run in result["runs"]))
                seal_predictions(config, output)
            self.assertTrue((output / "prediction_seal.json").is_file())
            final = evaluate_sealed(config, output, logger)
            self.assertEqual(set(final["summary"]), set(config["ablation"]["variants"]))
            self.assertEqual(final["test_images"], 2)
            audit = read_json(output / "reports" / "final_teacher_audit.json")
            self.assertIn("positive_pseudo_images", audit)
            self.assertNotIn("accepted_images", audit)
            self.assertFalse(final["manual_labels_used_for_training_or_selection"])


if __name__ == "__main__":
    unittest.main()
