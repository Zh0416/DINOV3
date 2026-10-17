import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from pvseg.label_free_data import RGBSample, rgb_hash
from pvseg.probe_coverage import probe_pseudo_coverage
from pvseg.teacher import parse_presence_response
from pvseg.utils import atomic_write_json, read_json


def fixture(root):
    image_path = root / "rgb.png"
    pixels = np.full((32, 32, 3), 224, np.uint8)
    pixels[:, :16] = [32, 64, 96]
    Image.fromarray(pixels).save(image_path)
    sample = RGBSample("synthetic", str(image_path), rgb_hash(image_path), "train")
    config = {
        "seed": 42, "protocol": {"mode": "soft_semantic"},
        "data": {"image_size": 32}, "dino": {"model_id": "synthetic-only"},
        "teacher": {"model_id": "synthetic-only", "prompt": "synthetic grid prompt"},
        "runtime": {"model_cache_dir": str(root / "model_cache")},
        "reliability": {"feature_size": 8, "grid_size": 4, "min_seed_pixels": 2,
                        "gaussian_sigma_patches": 0, "boundary_pixels": 1},
    }
    atomic_write_json(root / "rgb_manifest.json", {"samples": [sample.to_dict()]})
    atomic_write_json(root / "protocol_config.json", config)
    features = torch.zeros(2, 8, 8)
    features[1, :, :4] = 1
    features[0, :, 4:] = 1
    (root / "feature_cache").mkdir()
    torch.save(features, root / "feature_cache" / "synthetic.pt")
    views = {}
    for name in ("original", "hflip"):
        flipped = name == "hflip"
        views[name] = {"status": "ok", "flipped": flipped, "grid": [
            {"row": row, "col": col,
             "status": "positive" if (col >= 2 if flipped else col < 2) else "negative",
             "pv_probability": .9 if (col >= 2 if flipped else col < 2) else .1}
            for row in range(4) for col in range(4)
        ]}
    return sample, config, [{"sample_id": sample.sample_id, "protocols": {"grid": views}}]


class ProbeCoverageTests(unittest.TestCase):
    def test_real_fusion_and_target_builder_report_coverage_without_training_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample, config, records = fixture(root)
            before = {p.relative_to(root) for p in root.rglob("*")}
            report = probe_pseudo_coverage([sample], root, records, config)
            self.assertEqual(report["status"], "measured_diagnostic_only")
            self.assertGreater(report["train_class_pixels"]["positive"], 0)
            self.assertGreater(report["train_class_pixels"]["negative"], 0)
            self.assertGreater(report["effective_supervision_coverage"], 0)
            self.assertLess(report["effective_supervision_coverage"], 1)
            row = report["per_image"][0]
            self.assertEqual(row["teacher_positive_pixels"], 32)
            self.assertEqual(row["teacher_negative_pixels"], 32)
            self.assertEqual(before, {p.relative_to(root) for p in root.rglob("*")})

    def test_provenance_mismatch_is_rejected_before_feature_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample, config, records = fixture(root)
            changed = copy.deepcopy(config)
            changed["dino"]["model_id"] = "different-checkpoint"
            with patch("torch.load") as load:
                with self.assertRaisesRegex(ValueError, "configuration/image size"):
                    probe_pseudo_coverage([sample], root, records, changed)
                atomic_write_json(root / "rgb_manifest.json", {"samples": [
                    {**sample.to_dict(), "split": "test"}]})
                with self.assertRaisesRegex(ValueError, "hash/split mismatch"):
                    probe_pseudo_coverage([sample], root, records, config)
                load.assert_not_called()

    def test_missing_foreground_blocks_and_does_not_invent_final_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample, config, records = fixture(root)
            for view in records[0]["protocols"]["grid"].values():
                for cell in view["grid"]:
                    cell.update(status="negative", pv_probability=.1)
            with patch("torch.load") as load:
                report = probe_pseudo_coverage([sample], root, records, config)
                load.assert_not_called()
            self.assertEqual(report["status"], "blocked_prototype_seed_support")
            self.assertEqual(report["per_image"][0]["prototype_foreground_pixels_after_smoothing"], 0)
            self.assertIsNone(report["effective_supervision_coverage"])

    def test_conflicting_teacher_blocks_before_any_feature_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample, config, records = fixture(root)
            for name, view in records[0]["protocols"]["grid"].items():
                for cell in view["grid"]:
                    cell.update(status="positive" if name == "hflip" else "negative", pv_probability=.95)
            with patch("torch.load") as load:
                report = probe_pseudo_coverage([sample], root, records, config)
                load.assert_not_called()
            self.assertEqual(report["status"], "blocked_teacher_protocol")
            self.assertEqual(report["per_image"][0]["teacher_conflict_pixels"], 64)
            self.assertIsNone(report["effective_supervision_coverage"])

    def test_cli_synthetic_inference_saves_four_calls_and_read_only_coverage(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "12_teacher_protocol_probe.py"
        spec = importlib.util.spec_from_file_location("protocol_probe_cli", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config, records = fixture(root)
            config_path = root / "config.json"
            atomic_write_json(config_path, config)
            calls = []
            mode = {"conflict": False}

            class SyntheticTeacher:
                runtime_spec = {"model_reference": "synthetic-only", "quantization": "none"}

                def infer_presence(self, image_path, view, prompt):
                    calls.append(("presence", view))
                    return parse_presence_response('{"pv_probability":0.8,"status":"p"}')

                def infer(self, image_path, view, prompt):
                    calls.append(("grid", view))
                    result = copy.deepcopy(records[0]["protocols"]["grid"][view])
                    if mode["conflict"]:
                        for cell in result["grid"]:
                            cell.update(status="positive" if view == "hflip" else "negative",
                                        pv_probability=.95)
                    return result

            output = root / "probe"
            args = [str(script), "--config", str(config_path), "--manifest", str(root / "rgb_manifest.json"),
                    "--output-root", str(output), "--samples", "1", "--feature-root", str(root)]
            with patch("sys.argv", args), patch("torch.cuda.is_available", return_value=True), \
                    patch.object(module.InternVLTeacher, "load", return_value=SyntheticTeacher()):
                self.assertEqual(module.main(), 0)
            report = read_json(output / "teacher_protocol_probe.json")
            self.assertEqual(len(calls), 4)
            self.assertEqual(report["status"], "diagnostic_complete")
            self.assertEqual(report["conclusion"]["diagnosis"], "small_probe_passed")
            self.assertTrue(report["eligible_for_bounded_followup"])
            self.assertEqual(report["pseudo_coverage_dry_run"]["status"], "measured_diagnostic_only")
            self.assertFalse((output / "semantic_prompt.json").exists())
            self.assertFalse((output / "pseudo_cache").exists())
            self.assertFalse((root / "rgb_dino_prototypes.npy").exists())
            for protocols in report["records"][0]["protocols"].values():
                for view in protocols.values():
                    self.assertIn("generation_spec", view)
                    self.assertIn("generation_prompt", view)
            mode["conflict"] = True
            failed_output = root / "probe_blocked"
            args[args.index("--output-root") + 1] = str(failed_output)
            with patch("sys.argv", args), patch("torch.cuda.is_available", return_value=True), \
                    patch.object(module.InternVLTeacher, "load", return_value=SyntheticTeacher()):
                self.assertEqual(module.main(), 2)
            failed = read_json(failed_output / "teacher_protocol_probe.json")
            self.assertFalse(failed["eligible_for_bounded_followup"])
            self.assertEqual(failed["conclusion"]["diagnosis"], "grid_elicitation_instability")
            self.assertEqual(failed["pseudo_coverage_dry_run"]["status"], "blocked_teacher_protocol")
            self.assertIsNone(failed["pseudo_coverage_dry_run"]["effective_supervision_coverage"])


if __name__ == "__main__":
    unittest.main()
