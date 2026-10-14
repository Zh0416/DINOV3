import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from pvseg.data import discover_samples, find_dataset_root, make_stratified_split


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "pv-gf2"
        self.config = {
            "categories": ["正样本", "困难正样本", "负样本"],
            "image_dir_name": "images",
            "label_dir_names": ["labels"],
            "extensions": [".bmp"],
        }
        for category in self.config["categories"]:
            (self.root / category / "images").mkdir(parents=True)
            (self.root / category / "labels").mkdir(parents=True)
            for index in range(6):
                image = np.full((16, 16, 3), index, dtype=np.uint8)
                mask = np.full((16, 16), 255 if category != "负样本" else 0, dtype=np.uint8)
                Image.fromarray(image).save(self.root / category / "images" / f"x{index}.bmp")
                Image.fromarray(mask).save(self.root / category / "labels" / f"x{index}.bmp")

    def tearDown(self):
        self.temporary.cleanup()

    def test_discovery_and_stratified_split(self):
        found = find_dataset_root(Path(self.temporary.name), "pv-gf2", self.config["categories"])
        self.assertEqual(found, self.root)
        samples = discover_samples(found, self.config)
        self.assertEqual(len(samples), 18)
        selected = make_stratified_split(samples, 6, [0.5, 0.25, 0.25], seed=42)
        self.assertEqual(Counter(item.split for item in selected), {"train": 9, "val": 6, "test": 3})
        self.assertEqual(Counter(item.category for item in selected), Counter({name: 6 for name in self.config["categories"]}))


if __name__ == "__main__":
    unittest.main()

