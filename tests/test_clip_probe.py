import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from pvseg.clip_probe import (FrozenCLIPWindowTeacher, bounded_decision, classify_windows,
                              diagnose_pair, diagnose_window_pair, rasterize_regions, state_summary,
                              symmetric_starts, validate_settings, window_boxes)
from pvseg.config import load_config
from pvseg.label_free_data import RGBSample, rgb_hash
from pvseg.utils import atomic_write_json, read_json


REPO = Path(__file__).resolve().parents[1]


def settings():
    return load_config(REPO / "configs" / "clip_window_probe_v1.yaml")["clip_probe"]


class ClipRegionTests(unittest.TestCase):
    def test_windows_cover_edges_and_are_closed_under_flip_including_odd_spans(self):
        for length in (1, 127, 255, 256, 257, 511, 512, 513, 521, 600, 1024):
            size = min(length, 256)
            starts = symmetric_starts(length, size, min(128, size))
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1] + size, length)
            self.assertEqual(starts, sorted(length - size - x for x in starts))
            self.assertTrue(all(b - a <= size for a, b in zip(starts, starts[1:])))
        self.assertEqual(len(window_boxes(512, 512, settings())), 9)
        boxes = window_boxes(521, 512, settings())
        self.assertEqual(set(boxes), {(521-x1, y0, 521-x0, y1) for x0, y0, x1, y1 in boxes})
        with self.assertRaisesRegex(ValueError, "budget"):
            window_boxes(2048, 1024, settings())

    def test_cosine_margin_does_not_turn_absence_of_foreground_into_background(self):
        states, _ = classify_windows([[.5, .2, .1], [.1, .5, .2], [.1, .12, .11],
                                     [.51, .5, .1]], settings())
        np.testing.assert_array_equal(states, [1, 0, -1, -1])
        for invalid in ([[np.nan, .5]], [[1.5, .2]], [[.3]]):
            with self.assertRaises(ValueError):
                classify_windows(invalid, settings())

    def test_overlapping_opposite_votes_remain_unknown(self):
        state, conflict = rasterize_regions(6, 2, [(0, 0, 4, 2), (2, 0, 6, 2)], [1, 0])
        np.testing.assert_array_equal(state, [[1, 1, -1, -1, 0, 0]] * 2)
        self.assertEqual(int(conflict.sum()), 4)
        state2, _ = rasterize_regions(6, 2, [(0, 0, 4, 2), (2, 0, 6, 2)], [1, -1])
        self.assertTrue(np.all(state2[:, :4] == 1))
        self.assertTrue(np.all(state2[:, 4:] == -1))

    def test_pair_requires_two_views_and_empty_foreground_iou_is_null(self):
        original = np.array([[1, 1, 0, -1]], np.int8)
        flipped = np.array([[-1, 0, -1, 1]], np.int8)
        report, consensus = diagnose_pair(original, flipped)
        np.testing.assert_array_equal(consensus, [[1, -1, 0, -1]])
        self.assertEqual(report["conflict_fraction"], 0)
        self.assertEqual(report["consensus"]["valid_region_evidence_coverage"], .5)
        report, _ = diagnose_pair(np.full((2, 2), -1), np.full((2, 2), -1))
        self.assertEqual(report["transform_three_state_agreement"], 1)
        self.assertIsNone(report["joint_valid_agreement"])
        self.assertIsNone(report["positive_region_iou"])
        self.assertEqual(report["consensus"]["valid_region_evidence_coverage"], 0)
        report, _ = diagnose_pair(np.ones((2, 2)), np.zeros((2, 2)))
        self.assertEqual(report["conflict_fraction"], 1)
        self.assertEqual(report["consensus"]["valid_region_evidence_coverage"], 0)

    @staticmethod
    def window_view(size, boxes, labels, margins):
        return {"rgb_size": list(size), "windows": [
            {"box_xyxy": list(box), "state": label, "foreground_minus_best_background": margin}
            for box, label, margin in zip(boxes, labels, margins)]}

    def test_overlapping_regions_do_not_imply_same_location_window_repeatability(self):
        options = {**settings(), "window_size": 4, "stride": 2}
        boxes = window_boxes(8, 8, options)
        labels = [-1] * len(boxes)
        labels[boxes.index((2, 2, 6, 6))] = 0  # Only the center is known in the original.
        aligned = [-1] * len(boxes)
        aligned[boxes.index((0, 0, 4, 4))] = 0  # Only top-left is known in the aligned flip.
        flipped_boxes = [(8-x1, y0, 8-x0, y1) for x0, y0, x1, y1 in boxes]
        first = self.window_view((8, 8), boxes, labels, [-.04 if x == 0 else -.02 for x in labels])
        second = self.window_view((8, 8), flipped_boxes, aligned, [-.04 if x == 0 else -.02 for x in aligned])
        original_map, _ = rasterize_regions(8, 8, boxes, labels)
        flipped_map, _ = rasterize_regions(8, 8, flipped_boxes, aligned)
        region_report, _ = diagnose_pair(original_map, flipped_map)
        self.assertEqual(region_report["joint_valid_agreement"], 1)
        self.assertEqual(region_report["consensus"]["negative_region_coverage"], .0625)
        window_report = diagnose_window_pair(first, second)
        self.assertEqual(window_report["joint_valid_windows"], 0)
        self.assertEqual(window_report["repeat_negative_windows"], 0)
        self.assertEqual(window_report["known_unknown_changed_windows"], 2)
        self.assertIsNone(window_report["joint_valid_agreement"])
        self.assertIsNone(window_report["joint_valid_conflict_fraction"])
        self.assertAlmostEqual(window_report["transform_three_state_agreement"], 7/9)

    def test_window_pair_aligns_reflected_coordinates_independent_of_list_order(self):
        boxes = window_boxes(7, 5, {**settings(), "window_size": 3, "stride": 2})
        labels = [-1, 0, 1] * 2
        margins = [.01, -.07, .08] * 2
        first = self.window_view((7, 5), boxes, labels, margins)
        flipped_boxes = [(7-x1, y0, 7-x0, y1) for x0, y0, x1, y1 in boxes]
        second = self.window_view((7, 5), flipped_boxes, labels, margins)
        second["windows"].reverse()
        report = diagnose_window_pair(first, second)
        self.assertEqual(report["matched_windows"], 6)
        self.assertEqual(report["joint_valid_windows"], 4)
        self.assertEqual(report["repeat_positive_windows"], 2)
        self.assertEqual(report["repeat_negative_windows"], 2)
        self.assertEqual(report["transform_three_state_agreement"], 1)
        self.assertEqual(report["max_absolute_margin_change"], 0)
        second["windows"][0].update(state=0, foreground_minus_best_background=-.08)
        report = diagnose_window_pair(first, second)
        self.assertEqual(report["joint_valid_conflict_fraction"], .25)
        self.assertAlmostEqual(report["max_absolute_margin_change"], .16)

    def test_window_pair_rejects_ambiguous_or_invalid_geometry_and_scores(self):
        first = self.window_view((8, 4), [(0, 0, 4, 4), (4, 0, 8, 4)], [1, 0], [.1, -.1])
        for invalid in ("duplicate", "missing", "size", "empty", "nonfinite", "state", "bounds"):
            with self.subTest(invalid=invalid):
                second = copy.deepcopy(first)
                if invalid == "duplicate":
                    second["windows"].append(copy.deepcopy(second["windows"][0]))
                elif invalid == "missing":
                    second["windows"].pop()
                elif invalid == "size":
                    second["rgb_size"][0] = 9
                elif invalid == "empty":
                    second["windows"] = []
                elif invalid == "nonfinite":
                    second["windows"][0]["foreground_minus_best_background"] = float("nan")
                elif invalid == "state":
                    second["windows"][0]["state"] = 2
                else:
                    second["windows"][0]["box_xyxy"][0] = -1
                with self.assertRaises(ValueError):
                    diagnose_window_pair(first, second)

    def test_unknown_area_cannot_dilute_conflicts_in_the_followup_gate(self):
        original = np.full((10, 10), -1)
        aligned = original.copy()
        original[0, :3] = [1, 0, 1]
        aligned[0, :3] = [0, 1, 1]
        pair, _ = diagnose_pair(original, aligned[:, ::-1])
        self.assertEqual(pair["conflict_fraction"], .02)
        self.assertAlmostEqual(pair["joint_valid_conflict_fraction"], 2/3)
        row = {"pair": pair, "views": {"original": {"regions": state_summary(original)},
                                      "hflip": {"regions": state_summary(aligned)}}}
        result = bounded_decision([row], .25)
        self.assertIn("transform_conflict", result["reasons"])

    def test_uniform_or_unknown_output_cannot_pass_on_consistency_alone(self):
        for label in (-1, 0, 1):
            array = np.full((4, 4), label)
            pair, _ = diagnose_pair(array, array)
            row = {"pair": pair, "views": {"original": {"regions": state_summary(array)},
                                          "hflip": {"regions": state_summary(array)}}}
            result = bounded_decision([row], .25)
            self.assertFalse(result["eligible_for_bounded_followup"])
        invalid = copy.deepcopy(settings())
        invalid["backgrounds"]["photovoltaic"] = ["colliding concept"]
        with self.assertRaises(ValueError):
            validate_settings(invalid)


class ClipProbeIntegrationTests(unittest.TestCase):
    def test_real_transformers_tiny_random_clip_is_frozen_and_batched(self):
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("Optional transformers unavailable; no real CLIP semantic test")
        from transformers import CLIPConfig, CLIPModel

        class TinyProcessor:
            # Tests model APIs with synthetic tensors, never downloaded weights.
            def __call__(self, text=None, images=None, **kwargs):
                if text is not None:
                    ids = torch.tensor([[0, 4 + i % 16, 2] for i in range(len(text))])
                    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                tensors = [np.asarray(i.resize((16, 16)), dtype=np.float32).transpose(2, 0, 1) / 255 for i in images]
                return {"pixel_values": torch.from_numpy(np.stack(tensors))}

        config = CLIPConfig(projection_dim=16,
                            text_config={"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                                         "num_hidden_layers": 1, "num_attention_heads": 4,
                                         "max_position_embeddings": 8, "eos_token_id": 2},
                            vision_config={"hidden_size": 32, "intermediate_size": 64,
                                           "num_hidden_layers": 1, "num_attention_heads": 4,
                                           "image_size": 16, "patch_size": 8})
        model = CLIPModel(config)
        options = {**settings(), "window_size": 16, "stride": 8, "batch_size": 2}
        teacher = FrozenCLIPWindowTeacher(model, TinyProcessor(), options, {"synthetic": True})
        result, state = teacher.infer(Image.fromarray(np.zeros((24, 24, 3), np.uint8)))
        self.assertEqual(len(result["windows"]), 4)
        self.assertEqual(state.shape, (24, 24))
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertFalse(teacher.text_features.requires_grad)
        self.assertTrue(all(np.isfinite(list(w["cosine_scores"].values())).all() for w in result["windows"]))

    def test_cli_saves_diagnostics_rejects_mismatch_and_preserves_partial_failure(self):
        spec = importlib.util.spec_from_file_location("clip_probe_cli", REPO / "scripts" / "13_clip_window_probe.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "rgb.png"
            pixels = np.zeros((16, 16, 3), np.uint8)
            pixels[:, :8, 0] = 255
            pixels[:, 8:, 2] = 255
            Image.fromarray(pixels).save(image_path)
            sample = RGBSample("synthetic", str(image_path), rgb_hash(image_path), "train")
            atomic_write_json(root / "manifest.json", {"samples": [sample.to_dict()]})
            options = {**settings(), "window_size": 8, "stride": 8}
            atomic_write_json(root / "config.json", {"seed": 42, "runtime": {
                "model_cache_dir": str(root / "cache")}, "clip_probe": options})
            calls = []
            mode = {"uniform": False, "fail": False}

            class SyntheticTeacher:
                runtime_spec = {"model_reference": "synthetic-only"}

                def infer(self, image):
                    calls.append(image.size)
                    if mode["fail"] and len(calls) % 2 == 0:
                        raise RuntimeError("synthetic inference failure")
                    state = np.where(np.asarray(image)[:, :, 0] > 100, 1, 0).astype(np.int8)
                    if mode["uniform"]:
                        state[:] = 0
                    boxes = window_boxes(*image.size, options)
                    labels = [int(state[y0, x0]) for x0, y0, _, _ in boxes]
                    windows = ClipRegionTests.window_view(image.size, boxes, labels,
                                                          [.1 if x == 1 else -.1 for x in labels])
                    return {"status": "ok", "regions": state_summary(state), **windows,
                            "elapsed_seconds": .1}, state

            args = ["13_clip_window_probe.py", "--config", str(root / "config.json"),
                    "--manifest", str(root / "manifest.json"), "--output-root", str(root / "probe")]
            with patch("sys.argv", args), patch.object(module.FrozenCLIPWindowTeacher, "load", return_value=SyntheticTeacher()):
                self.assertEqual(module.main(), 0)
            report = read_json(root / "probe" / "clip_window_probe.json")
            self.assertEqual(report["status"], "diagnostic_complete")
            self.assertEqual(len(calls), 2)
            self.assertIsNone(report["effective_supervision_coverage"])
            self.assertEqual(report["records"][0]["pair"]["positive_region_iou"], 1)
            self.assertEqual(report["records"][0]["window_pair"]["repeat_positive_windows"], 2)
            self.assertEqual(report["records"][0]["window_pair"]["repeat_negative_windows"], 2)
            self.assertTrue((root / "probe" / "synthetic_regions.png").is_file())
            self.assertFalse((root / "probe" / "semantic_prompt.json").exists())
            self.assertFalse((root / "probe" / "pseudo_cache").exists())

            args[-1] = str(root / "blocked")
            mode["uniform"] = True
            with patch("sys.argv", args), patch.object(module.FrozenCLIPWindowTeacher, "load", return_value=SyntheticTeacher()):
                self.assertEqual(module.main(), 2)
            mode["fail"] = True
            args[-1] = str(root / "failure")
            with patch("sys.argv", args), patch.object(module.FrozenCLIPWindowTeacher, "load", return_value=SyntheticTeacher()):
                with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
                    module.main()
            failed = read_json(root / "failure" / "clip_window_probe.json")
            self.assertEqual(failed["status"], "technical_failure")
            self.assertEqual(list(failed["records"][0]["views"]), ["original"])

            atomic_write_json(root / "previous.json", {"status": "diagnostic_complete", "manual_labels_used": False,
                                                       "records": [{"sample_id": "wrong", "rgb_sha256": "wrong"}]})
            args[-1] = str(root / "mismatch")
            with patch("sys.argv", args + ["--compare-report", str(root / "previous.json")]), \
                    patch.object(module.FrozenCLIPWindowTeacher, "load") as load:
                with self.assertRaisesRegex(ValueError, "IDs/order/hashes"):
                    module.main()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
