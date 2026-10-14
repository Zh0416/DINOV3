import unittest

import numpy as np

from pvseg.attention import gaussian_attention_map, sanitize_boxes


class AttentionTests(unittest.TestCase):
    def test_gaussian_attention_has_expected_shapes(self):
        full, reduced = gaussian_attention_map(
            [[100, 200, 300, 400]], image_size=1024, feature_size=64, sigma=32, confidence=0.8
        )
        self.assertEqual(full.shape, (1024, 1024))
        self.assertEqual(reduced.shape, (64, 64))
        self.assertGreater(float(reduced.max()), 0.7)
        self.assertLessEqual(float(reduced.max()), 0.8 + 1e-5)

    def test_invalid_boxes_are_removed(self):
        boxes = sanitize_boxes([[5, 5, 5, 20], [-5, 1, 10, 12], [1, 2, 3]])
        self.assertEqual(boxes, [[0, 1, 10, 12]])
        full, reduced = gaussian_attention_map([], image_size=64, feature_size=4, sigma=2)
        self.assertFalse(np.any(full))
        self.assertFalse(np.any(reduced))


if __name__ == "__main__":
    unittest.main()

