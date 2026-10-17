import unittest

import numpy as np
from PIL import Image

from pvseg.teacher import parse_presence_response
from pvseg.teacher.internvl import _dynamic_preprocess
from pvseg.teacher_probe import (DEFAULT_PRESENCE_PROMPT, classify_probe,
                                 diagnose_grid_pair, diagnose_presence_pair)


def grid_view(status, probability, flipped=False):
    return {
        "status": "ok", "flipped": flipped, "patch_count": 1,
        "grid": [
            {"row": row, "col": col, "status": status, "pv_probability": probability}
            for row in range(4) for col in range(4)
        ],
    }


class TeacherProtocolProbeTests(unittest.TestCase):
    def test_default_presence_prompt_defines_probability_not_confidence(self):
        self.assertIn("pv_probability", DEFAULT_PRESENCE_PROMPT)
        self.assertIn("不是回答自信度", DEFAULT_PRESENCE_PROMPT)

    def test_presence_parser_is_strict_and_keeps_generated_probability(self):
        parsed = parse_presence_response('```json\n{"pv_probability":0.73,"status":"p"}\n```')
        self.assertTrue(parsed["parse_valid"])
        self.assertEqual(parsed["status_label"], "positive")
        self.assertEqual(parsed["pv_probability"], .73)
        invalid = parse_presence_response('{"status":"yes"}')
        self.assertFalse(invalid["parse_valid"])
        self.assertEqual(invalid["status_label"], "unknown")

    def test_presence_invalid_numbers_never_become_valid_negative_evidence(self):
        for value in ('NaN', 'Infinity', '95', '-1', 'true', 'null', '"bad"'):
            with self.subTest(value=value):
                result = parse_presence_response('{"pv_probability":' + value + ',"status":"n"}')
                self.assertFalse(result["parse_valid"])

    def test_grid_probe_exposes_flip_conflict_and_probability_semantics(self):
        result = diagnose_grid_pair({
            "original": grid_view("negative", .95),
            "hflip": grid_view("positive", .95, flipped=True),
        })
        self.assertEqual(result["conflict_fraction"], 1.)
        self.assertEqual(result["all_negative_views"], 1)
        self.assertEqual(result["all_positive_views"], 1)
        self.assertEqual(result["probability_status_violations"], 16)
        self.assertEqual(result["pipeline_effective_support_coverage"], 0.)
        self.assertEqual(result["protocol_valid_support_coverage"], 0.)

    def test_grid_probe_separates_pipeline_and_protocol_valid_coverage(self):
        result = diagnose_grid_pair({
            "original": grid_view("negative", .95),
            "hflip": grid_view("negative", .95, flipped=True),
        })
        self.assertEqual(result["pipeline_effective_support_coverage"], 1.)
        self.assertEqual(result["protocol_valid_support_coverage"], 0.)
        self.assertEqual(result["probability_status_violations"], 32)

    def test_probe_classification_does_not_turn_presence_into_pixel_labels(self):
        stable_presence = diagnose_presence_pair({
            "original": {"parse_valid": True, "status_label": "positive", "pv_probability": .8},
            "hflip": {"parse_valid": True, "status_label": "positive", "pv_probability": .8},
        })
        unstable_grid = diagnose_grid_pair({
            "original": grid_view("negative", .1),
            "hflip": grid_view("positive", .9, flipped=True),
        })
        conclusion = classify_probe([{"presence": stable_presence, "grid": unstable_grid}], .25)
        self.assertEqual(conclusion["diagnosis"], "grid_elicitation_instability")
        self.assertIn("replace_or_redesign", conclusion["decision"])

    def test_consistent_unknown_presence_is_not_a_pass(self):
        presence = diagnose_presence_pair({
            "original": {"parse_valid": True, "status_label": "unknown", "pv_probability": .4},
            "hflip": {"parse_valid": True, "status_label": "unknown", "pv_probability": .4},
        })
        grid = diagnose_grid_pair({
            "original": grid_view("unknown", .4),
            "hflip": grid_view("unknown", .4, flipped=True),
        })
        self.assertEqual(grid["protocol_valid_support_coverage"], 0.)
        conclusion = classify_probe([{"presence": presence, "grid": grid}], .25)
        self.assertEqual(conclusion["diagnosis"], "global_presence_abstained")

    def test_incomplete_and_unknown_grids_do_not_pass_with_stable_presence(self):
        presence = diagnose_presence_pair({name: {
            "parse_valid": True, "status_label": "positive", "pv_probability": .8,
        } for name in ("original", "hflip")})
        unknown = grid_view("unknown", .4)
        grid = diagnose_grid_pair({"original": unknown, "hflip": unknown})
        self.assertEqual(classify_probe([{"presence": presence, "grid": grid}], .25)["diagnosis"],
                         "no_usable_spatial_evidence")
        grid = diagnose_grid_pair({"original": unknown, "hflip": {"status": "error"}})
        self.assertEqual(classify_probe([{"presence": presence, "grid": grid}], .25)["diagnosis"],
                         "technical_or_output_schema_failure")

    def test_stable_but_contradictory_protocols_do_not_pass(self):
        presence = diagnose_presence_pair({name: {
            "parse_valid": True, "status_label": "positive", "pv_probability": .8,
        } for name in ("original", "hflip")})
        grid = diagnose_grid_pair({name: grid_view("negative", .1)
                                   for name in ("original", "hflip")})
        self.assertEqual(classify_probe([{"presence": presence, "grid": grid}], .25)["diagnosis"],
                         "presence_grid_disagreement")

    def test_square_preprocessing_is_exactly_flip_reversible(self):
        pixels = np.zeros((512, 512, 3), np.uint8)
        pixels[:, :180] = [240, 10, 30]
        pixels[100:300, 300:] = [20, 210, 70]
        image = Image.fromarray(pixels)
        original = _dynamic_preprocess(image, 448, 4, use_thumbnail=True)
        flipped = _dynamic_preprocess(
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), 448, 4, use_thumbnail=True
        )
        self.assertEqual(len(original), 1)
        self.assertEqual(len(flipped), 1)
        np.testing.assert_array_equal(np.asarray(flipped[0]), np.fliplr(np.asarray(original[0])))

    def test_large_square_tile_order_is_known_under_flip(self):
        pixels = np.arange(1024 * 1024, dtype=np.uint32).reshape(1024, 1024)
        pixels = np.repeat((pixels % 251).astype(np.uint8)[..., None], 3, axis=2)
        image = Image.fromarray(pixels)
        original = _dynamic_preprocess(image, 448, 4, use_thumbnail=True)
        flipped = _dynamic_preprocess(
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), 448, 4, use_thumbnail=True
        )
        self.assertEqual(len(original), 5)
        for source, destination in ((1, 0), (0, 1), (3, 2), (2, 3), (4, 4)):
            np.testing.assert_array_equal(
                np.asarray(flipped[destination]), np.fliplr(np.asarray(original[source]))
            )


if __name__ == "__main__":
    unittest.main()
