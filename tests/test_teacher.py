import json
import unittest

from pvseg.teacher import parse_teacher_response


class TeacherParserTests(unittest.TestCase):
    def test_strict_json(self):
        result = parse_teacher_response(
            '```json\n{"pv_exist": true, "confidence": 95, "bbox": [[10, 20, 30, 40]], "reason": "rows"}\n```'
        )
        self.assertTrue(result["pv_exist"])
        self.assertAlmostEqual(result["confidence"], 0.95)
        self.assertEqual(result["bbox"], [[10, 20, 30, 40]])

    def test_negative_clears_bbox(self):
        result = parse_teacher_response(
            '{"pv_exist": false, "confidence": 0.9, "bbox": [[1, 2, 3, 4]], "reason": "none"}'
        )
        self.assertFalse(result["pv_exist"])
        self.assertEqual(result["bbox"], [])

    def test_grid_teacher_does_not_require_bbox(self):
        result = parse_teacher_response(
            '{"pv_exist": true, "image_confidence": 0.8, '
            '"grid": [[{"pv_probability": 0.9, "status": "positive"}]]}'
        )
        self.assertEqual(len(result["grid"]), 1)
        self.assertNotEqual(result.get("parse_warning"), "pv_exists_without_valid_bbox")

    def test_truncated_grid_recovers_complete_cells(self):
        raw = (
            '```json\n{"pv_exist":true,"image_confidence":0.9,"grid":['
            '[{"row":0,"col":0,"pv_probability":0.9,"status":"positive"},'
            '{"row":0,"col":1,"pv_probability":0.8,"status":"positive"},'
            '{"row":0,"col":2,"pv_probability":0.1,"status":"unknown"},'
            '{"row":0,"col":3,"pv_probability":'
        )
        result = parse_teacher_response(raw)
        self.assertEqual(len(result["grid"]), 3)
        self.assertEqual(result["grid"][0]["status"], "positive")
        self.assertEqual(result.get("parse_warning"), "truncated_json_recovered_grid")

    def test_coordinate_convention_is_applied_to_the_whole_grid(self):
        for offset in (0, 1):
            cells = [{"row": row + offset, "col": col + offset,
                      "pv_probability": (row * 4 + col) / 100., "status": "positive"}
                     for row in range(4) for col in range(4)]
            complete = json.dumps({"pv_exist": True, "grid": cells})
            for raw in (complete, complete[:-2], '{"grid":' + json.dumps(cells[:8])[:-1]):
                with self.subTest(offset=offset, truncated=raw != complete, cells=len(raw)):
                    grid = parse_teacher_response(raw)["grid"]
                    expected = 8 if raw.startswith('{"grid":') else 16
                    self.assertEqual(len(grid), expected)
                    for index, item in enumerate(grid):
                        self.assertEqual((item["row"], item["col"]), divmod(index, 4))
                        self.assertAlmostEqual(item["pv_probability"], index / 100.)

    def test_ambiguous_partial_coordinates_follow_zero_based_prompt(self):
        grid = parse_teacher_response('{"grid":[{"row":1,"col":2,"pv_probability":0.8}')["grid"]
        self.assertEqual((grid[0]["row"], grid[0]["col"]), (1, 2))


if __name__ == "__main__":
    unittest.main()

