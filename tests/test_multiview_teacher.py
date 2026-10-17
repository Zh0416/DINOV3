import copy
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pvseg.multiview_teacher import generate_grid_teacher, grid_generation_spec, global_pair_diagnostic
from pvseg.utils import atomic_write_json, read_json


class GridTeacherResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {"teacher": {"model_id": "test-teacher", "prompt": "new prompt", "local_views": False},
                       "runtime": {"resume": True}, "reliability": {"grid_size": 4}}
        atomic_write_json(self.root / "rgb_manifest.json", {"samples": [
            {"sample_id": "sample", "image_path": "unused", "rgb_sha256": "sample", "split": "train"}]})
        self.view = {"status": "ok", "grid": [], "parse_warning": "pv_exists_without_valid_bbox",
                     "raw_response": '{"pv_exist":true,"grid":[{"row":0,"col":1,"pv_probability":0.9,"status":"positive"}',
                     "crop_box": [0, 0, 563, 563], "flipped": True, "patch_count": 2}

    def tearDown(self):
        self.temporary.cleanup()

    def cache(self, flipped=None, model="test-teacher", historical=False):
        views = {"original": copy.deepcopy(self.view),
                 "hflip": copy.deepcopy(self.view if flipped is None else flipped)}
        if not historical:
            for view in views.values():
                view.update({"generation_prompt": "new prompt", "teacher_model": model,
                             "generation_spec": grid_generation_spec(self.config)})
        atomic_write_json(self.root / "semantic_prompt.json", {
            "teacher_model": model, "prompt": "old prompt" if historical else "new prompt",
            "generation_spec": None if historical else grid_generation_spec(self.config),
            "records": [{"sample_id": "sample", "views": views}]})

    def test_fully_cached_resume_reparses_and_persists_without_cuda_or_model(self):
        self.cache(historical=True)
        self.config["teacher"]["cache_only_replay"] = True
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load, patch("pvseg.multiview_teacher.torch.cuda.is_available") as cuda:
            report = generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        load.assert_not_called()
        cuda.assert_not_called()
        self.assertEqual(report["new_inferences"], 0)
        self.assertEqual(report["reparsed_views"], 2)
        self.assertEqual(report["reused_views"], 2)
        saved = read_json(self.root / "semantic_prompt.json")["records"][0]["views"]["original"]
        self.assertEqual(saved["grid"][0]["col"], 1)
        self.assertEqual(saved["crop_box"], self.view["crop_box"])
        self.assertTrue(saved["flipped"])
        self.assertEqual(saved["patch_count"], 2)
        self.assertEqual(saved["generation_prompt"], "old prompt")
        self.assertEqual(saved["parse_warning"], "truncated_json_recovered_grid")

    def test_only_failed_view_is_retried(self):
        self.cache(flipped={"status": "error", "error": "previous failure", "grid": []})
        self.config["runtime"]["model_cache_dir"] = str(self.root / "models")
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load, patch("pvseg.multiview_teacher.torch.cuda.is_available", return_value=True):
            load.return_value.infer.return_value = {"grid": [{"row": 0, "col": 0, "pv_probability": .8, "status": "positive"}]}
            report = generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        self.assertEqual(report["new_inferences"], 1)
        self.assertEqual(report["reused_views"], 1)
        load.return_value.infer.assert_called_once_with("unused", view="hflip", crop_box=None, prompt="new prompt")
        saved = read_json(self.root / "semantic_prompt.json")["records"][0]["views"]
        self.assertEqual(saved["original"]["generation_prompt"], "new prompt")
        self.assertEqual(saved["hflip"]["generation_prompt"], "new prompt")

    def test_strict_cached_flat_grid_is_reparsed_without_inference(self):
        self.config["teacher"]["require_complete_grid"] = True
        self.view["raw_response"] = json.dumps({"pv_exist": False, "grid": [[.95, "n"]] * 16})
        self.view.update({"grid_cell_count": 0, "grid_complete": False, "grid_parser_version": 3})
        self.cache()
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load, patch("pvseg.multiview_teacher.torch.cuda.is_available") as cuda:
            report = generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        load.assert_not_called()
        cuda.assert_not_called()
        self.assertEqual(report["new_inferences"], 0)
        self.assertEqual(report["complete_grid_views"], 2)
        self.assertEqual(report["grid_cells"], 32)
        saved = read_json(self.root / "semantic_prompt.json")["records"][0]["views"]["original"]
        self.assertTrue(saved["grid_complete"])
        self.assertNotIn("parse_warning", saved)
        self.assertEqual(saved["crop_box"], self.view["crop_box"])

    def test_teacher_model_mismatch_does_not_silently_reuse_or_rerun(self):
        self.cache(model="different-teacher")
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load:
            with self.assertRaisesRegex(ValueError, "teacher model differs"):
                generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        load.assert_not_called()

    def test_incomplete_cached_grid_stops_without_inference(self):
        self.cache()
        self.config["teacher"]["require_complete_grid"] = True
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load:
            with self.assertRaisesRegex(RuntimeError, "Incomplete cached teacher grid"):
                generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        load.assert_not_called()
        saved = read_json(self.root / "semantic_prompt.json")["records"][0]["views"]["original"]
        self.assertEqual(saved["grid_cell_count"], 1)
        self.assertFalse(saved["grid_complete"])

    def test_incomplete_new_grid_is_saved_and_stops_after_first_inference(self):
        self.config["teacher"]["require_complete_grid"] = True
        self.config["runtime"]["model_cache_dir"] = str(self.root / "models")
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load, patch("pvseg.multiview_teacher.torch.cuda.is_available", return_value=True):
            load.return_value.infer.return_value = {"raw_response": self.view["raw_response"],
                                                    "grid": [{"row": 0, "col": 0, "status": "positive", "pv_probability": .8}]}
            with self.assertRaisesRegex(RuntimeError, "Incomplete teacher grid"):
                generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        self.assertEqual(load.return_value.infer.call_count, 1)
        saved = read_json(self.root / "semantic_prompt.json")["records"][0]["views"]
        self.assertEqual(set(saved), {"original"})
        self.assertIn("raw_response", saved["original"])

    def test_changed_prompt_or_generation_settings_refuse_cache_without_inference(self):
        self.cache()
        for key, value in (("prompt", "another prompt"), ("max_new_tokens", 512), ("dtype", "bfloat16"),
                           ("quantization", "none"), ("use_thumbnail", True)):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config["teacher"][key] = value
                with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load:
                    with self.assertRaisesRegex(ValueError, "generation settings differ"):
                        generate_grid_teacher(config, self.root, logging.getLogger(__name__))
                load.assert_not_called()

    def test_historical_replay_cannot_retry_failed_view(self):
        self.cache(flipped={"status": "error", "grid": []}, historical=True)
        self.config["teacher"]["cache_only_replay"] = True
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load:
            with self.assertRaisesRegex(RuntimeError, "Cache-only replay cannot infer"):
                generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        load.assert_not_called()

    def test_mixed_per_view_prompt_is_rejected_even_if_top_level_spec_matches(self):
        self.cache()
        payload = read_json(self.root / "semantic_prompt.json")
        payload["records"][0]["views"]["hflip"]["generation_prompt"] = "old prompt"
        atomic_write_json(self.root / "semantic_prompt.json", payload)
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load:
            with self.assertRaisesRegex(ValueError, "mixed teacher evidence"):
                generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        load.assert_not_called()

    def test_auto_quantization_cannot_silently_change_effective_precision_on_resume(self):
        self.cache(flipped={"status": "error", "grid": []})
        payload = read_json(self.root / "semantic_prompt.json")
        payload["records"][0]["views"]["original"]["runtime_spec"] = {"quantization": "8bit", "dtype": "torch.float16"}
        atomic_write_json(self.root / "semantic_prompt.json", payload)
        self.config["runtime"]["model_cache_dir"] = str(self.root / "models")
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load, patch(
                "pvseg.multiview_teacher.torch.cuda.is_available", return_value=True):
            load.return_value.runtime_spec = {"quantization": "4bit", "dtype": "torch.float16"}
            with self.assertRaisesRegex(RuntimeError, "effective precision/model differs"):
                generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        load.return_value.infer.assert_not_called()

    def test_global_pair_conflict_stops_before_local_inference_and_saves_raw(self):
        self.config["teacher"].update({"local_views": True, "require_complete_grid": True})
        self.config["runtime"]["model_cache_dir"] = str(self.root / "models")
        def response(status):
            raw = json.dumps({"grid": [[.95, status]] * 16})
            from pvseg.teacher import parse_teacher_response
            return parse_teacher_response(raw)
        with patch("pvseg.multiview_teacher.InternVLTeacher.load") as load, patch(
                "pvseg.multiview_teacher.torch.cuda.is_available", return_value=True):
            load.return_value.infer.side_effect = [response("n"), response("p")]
            with self.assertRaisesRegex(RuntimeError, "original/hflip conflict on 16/16"):
                generate_grid_teacher(self.config, self.root, logging.getLogger(__name__))
        self.assertEqual(load.return_value.infer.call_count, 2)
        saved = read_json(self.root / "semantic_prompt.json")["records"][0]["views"]
        self.assertEqual(set(saved), {"original", "hflip"})
        self.assertIn("raw_response", saved["hflip"])
        checks = read_json(self.root / "reports/teacher_pair_checks.json")
        self.assertEqual(checks["status"], "failed")
        self.assertEqual(checks["per_image"][0]["conflict_fraction"], 1.)

    def test_pair_check_restores_hflip_coordinates(self):
        original = {"grid": [{"row": 0, "col": 0, "status": "positive", "pv_probability": .9}]}
        flipped = {"flipped": True, "grid": [{"row": 0, "col": 3, "status": "positive", "pv_probability": .9}]}
        result = global_pair_diagnostic({"original": original, "hflip": flipped})
        self.assertEqual(result["observed_pairs"], 1)
        self.assertEqual(result["conflicting_cells"], [])
