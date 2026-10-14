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


if __name__ == "__main__":
    unittest.main()

