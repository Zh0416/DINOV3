"""Regression tests for unsafe weak-supervision shortcuts; synthetic data only."""
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from pvseg.label_free_data import RGBSample
from pvseg.label_free_training import train_label_free
from pvseg.multiview_teacher import grid_generation_spec
from pvseg.soft_semantic import (SOFT_PSEUDO_VERSION, build_soft_pseudo_targets,
                                 fuse_soft_attention, generate_soft_pseudo_cache)
from pvseg.teacher.internvl import _dynamic_preprocess
from pvseg.utils import atomic_write_json, read_json


def grid_view(positive=True, score=.95):
    return {"status": "ok", "grid": [
        {"row": r, "col": c, "pv_probability": score, "status": "positive" if positive else "negative"}
        for r in range(4) for c in range(4)]}


class SoftSafetyTests(unittest.TestCase):
    def test_coarse_positive_grid_cannot_override_background_appearance(self):
        size = 8
        features = np.zeros((2, size, size), np.float32)
        features[0] = 1
        record = {"views": {"original": grid_view(), "second": grid_view()}}
        with patch("pvseg.soft_semantic.rgb_seeds", return_value=(np.zeros((size, size), bool), np.ones((size, size), bool))):
            arrays, diagnostic = build_soft_pseudo_targets(features, "unused", np.eye(2), record,
                {"feature_size": size, "boundary_pixels": 0, "gaussian_sigma_patches": 0})
        self.assertEqual(diagnostic["positive_pixels"], 0)
        self.assertEqual(diagnostic["appearance_rejected_pixels"], 64)
        self.assertTrue((arrays["target_soft"] == 255).all())

    def test_grid_existence_selects_only_supported_spatial_seeds(self):
        size = 8
        fg = np.zeros((size, size), bool)
        fg[2:4, 2:4] = True
        for rgb_support, dino_support, expected in ((True, True, 4), (True, False, 0), (False, True, 0)):
            with self.subTest(rgb=rgb_support, dino=dino_support):
                features = np.zeros((2, size, size), np.float32)
                features[0] = 1
                if dino_support:
                    features[0, fg], features[1, fg] = 0, 1
                rgb = fg if rgb_support else np.zeros_like(fg)
                with patch("pvseg.soft_semantic.rgb_seeds", return_value=(rgb, ~rgb)):
                    arrays, diag = build_soft_pseudo_targets(features, "unused", np.eye(2),
                        {"views": {"original": grid_view(), "second": grid_view()}},
                        {"feature_size": size, "boundary_pixels": 0, "gaussian_sigma_patches": 0})
                self.assertEqual(diag["positive_pixels"], expected)
                self.assertEqual(int(arrays["teacher_candidate"].sum()), expected)

    def test_single_view_abstains_and_multi_view_fusion_uses_mean(self):
        settings = {"feature_size": 8, "gaussian_sigma_patches": 0}
        attention, diag = fuse_soft_attention({"views": {"one": grid_view()}}, settings)
        np.testing.assert_allclose(attention, .2)
        self.assertTrue(diag["teacher_abstained"])
        attention, _ = fuse_soft_attention({"views": {"one": grid_view(), "two": grid_view(score=.56)}}, settings)
        np.testing.assert_allclose(attention, .755)

    def test_unknown_teacher_region_cannot_become_machine_background(self):
        features = np.zeros((2, 8, 8), np.float32)
        features[0] = 1
        with patch("pvseg.soft_semantic.rgb_seeds", return_value=(np.zeros((8, 8), bool), np.ones((8, 8), bool))):
            arrays, _ = build_soft_pseudo_targets(features, "unused", np.eye(2), {"views": {}}, {"feature_size": 8})
        self.assertTrue((arrays["target_soft"] == 255).all())

    def test_prototypes_exclude_unknown_color_seeds_and_all_test_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = {"feature_size": 8, "min_seed_pixels": 1, "boundary_pixels": 0,
                        "gaussian_sigma_patches": 0}
            config = {"teacher": {"model_id": "synthetic", "prompt": "grid"}, "reliability": settings}
            samples = [RGBSample(name, str(root / f"{name}.png"), name, split)
                       for name, split in (("train", "train"), ("test", "test"))]
            (root / "feature_cache").mkdir()
            records = []
            for sample in samples:
                rgb = np.full((8, 8, 3), 230, np.uint8)
                rgb[:, :4] = [35, 40, 65]
                Image.fromarray(rgb).save(sample.image_path)
                features = torch.zeros(4, 8, 8)
                features[0, :, 4:] = 1
                features[1, :, :2] = 1
                features[2, :, 2:4] = 1  # Water-like color in unknown teacher region.
                if sample.split == "test":
                    features[:] = 0
                    features[3] = 1  # Would contaminate prototypes if test were fitted.
                torch.save(features, root / "feature_cache" / f"{sample.sample_id}.pt")
                view = grid_view()
                view.update({"teacher_model": "synthetic", "generation_prompt": "grid",
                             "generation_spec": grid_generation_spec(config)})
                for cell in view["grid"]:
                    cell["status"] = ("positive", "unknown", "negative", "negative")[cell["col"]]
                records.append({"sample_id": sample.sample_id, "views": {"one": view, "two": view}})
            atomic_write_json(root / "rgb_manifest.json", {"samples": [s.to_dict() for s in samples]})
            atomic_write_json(root / "semantic_prompt.json", {"records": records, "teacher_model": "synthetic",
                                                               "generation_spec": grid_generation_spec(config)})
            generate_soft_pseudo_cache(config, root, logging.getLogger(__name__))
            np.testing.assert_allclose(np.load(root / "rgb_dino_prototypes.npy"), np.eye(4)[:2])
            provenance = read_json(root / "reports/prototype_provenance.json")
            self.assertEqual(provenance["train_ids"], ["train"])
            report = read_json(root / "reports/pseudo_reliability.json")
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["rejection_counts"], {"positive_and_background": 1, "abstained": 1})

    def test_direct_training_rejects_stale_failed_single_class_or_nonfinite_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = [RGBSample("s", "unused", "s", "train")]
            atomic_write_json(root / "rgb_manifest.json", {"samples": [s.to_dict() for s in samples]})
            (root / "pseudo_cache").mkdir()
            settings = {"feature_size": 4}
            config = {"protocol": {"mode": "soft_semantic", "label_usage": "final_evaluation_only"},
                      "training": {}, "reliability": settings}
            for case in ("stale", "failed", "single_class", "nonfinite"):
                with self.subTest(case=case):
                    target = np.zeros((4, 4), np.uint8)
                    if case != "single_class":
                        target[0, 0] = 1
                    weight = np.ones((4, 4), np.float32)
                    if case == "nonfinite":
                        weight[0, 0] = np.nan
                    np.savez(root / "pseudo_cache/s.npz", pseudo_version=SOFT_PSEUDO_VERSION,
                             target_soft=target, weight_soft=weight, attention_soft=np.full((4, 4), .2))
                    atomic_write_json(root / "reports/pseudo_reliability.json", {
                        "status": "failed" if case == "failed" else "ready",
                        "implementation_version": SOFT_PSEUDO_VERSION - int(case == "stale"),
                        "settings": settings, "per_image": [{"sample_id": "s", "split": "train"}]})
                    with patch("pvseg.label_free_training.CachedFeatureSegmenter") as model:
                        with self.assertRaises(RuntimeError):
                            train_label_free(config, root, logging.getLogger(__name__), device="cpu")
                    model.assert_not_called()

    def test_thumbnail_retains_global_context_alongside_spatial_tiles(self):
        rgb = np.zeros((1024, 1024, 3), np.uint8)
        rgb[:512, :512] = [255, 0, 0]
        rgb[512:, 512:] = [0, 0, 255]
        tiles = _dynamic_preprocess(Image.fromarray(rgb), 448, 4, use_thumbnail=True)
        self.assertEqual(len(tiles), 5)
        self.assertTrue(all(tile.size == (448, 448) for tile in tiles))
        thumbnail = np.asarray(tiles[-1])
        self.assertEqual(int(thumbnail[100, 100, 0]), 255)
        self.assertEqual(int(thumbnail[350, 350, 2]), 255)


if __name__ == "__main__":
    unittest.main()
