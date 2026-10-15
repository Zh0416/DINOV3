import copy
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pvseg.final_evaluation import evaluate_sealed
from pvseg.label_free_data import (RGBSample, PseudoFeatureDataset, manual_label_firewall,
                                   prepare_rgb_manifest, load_rgb_manifest)
from pvseg.label_free_training import train_label_free, VARIANTS, shuffle_donors
from pvseg.losses import PartialPseudoLoss
from pvseg.multiview_teacher import restore_hflip_boxes
from pvseg.prediction_seal import seal_predictions, verify_seal
from pvseg.pseudo_labels import fit_prototypes, generate_pseudo_cache, view_agreement, build_pseudo_targets
from pvseg.teacher import parse_teacher_response
from pvseg.utils import atomic_write_json


class LabelFreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "outputs"
        self.output.mkdir()
        self.logger = logging.getLogger("synthetic_label_free")
        self.settings = {"feature_size": 4, "view_iou_min": .5, "box_area_min": .0005,
                         "box_area_max": .6, "min_seed_pixels": 2, "feature_margin": .04,
                         "gaussian_sigma_patches": 1.}
        self.config = {
            "seed": 20261015, "protocol": {"label_usage": "final_evaluation_only", "name": "synthetic"},
            "data": {"input_root": str(self.root), "dataset_slug": "pv-gf2",
                     "categories": ["正样本", "困难正样本", "负样本"], "image_dir_name": "images",
                     "label_dir_names": ["labels"], "extensions": [".bmp"], "image_size": 16,
                     "max_samples": 6, "train_fraction": 2/3, "rgb_staging_dir": str(self.root / "staged")},
            "dino": {"expected_hidden_dim": 8}, "reliability": self.settings,
            "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "decoder_channels": 32,
                         "adapter_bottleneck": 4, "learning_rate": .001, "weight_decay": .0001,
                         "dice_weight": 1., "cross_entropy_weight": 1., "consistency_weight": .1,
                         "threshold": .5, "visualization_samples": 1},
            "ablation": {"seeds": [42]}, "runtime": {"output_dir": str(self.output)}}
        for cat_index, category in enumerate(self.config["data"]["categories"]):
            image_dir = self.root / "pv-gf2" / category / "images"
            label_dir = image_dir.parent / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir()
            for index in range(3):
                rgb = np.full((16, 16, 3), 230, np.uint8)
                rgb[:, :8] = [35, 40, 65]
                rgb[0, 0, 0] = 60 + cat_index*10 + index
                Image.fromarray(rgb).save(image_dir / f"{cat_index}_{index}.bmp")
                # These synthetic masks intentionally need not match directory labels.
                mask = np.zeros((16, 16), np.uint8)
                mask[:, :8] = 255
                Image.fromarray(mask).save(label_dir / f"{cat_index}_{index}.bmp")

    def tearDown(self):
        self.temp.cleanup()

    def record(self):
        view = {"status": "ok", "pv_exist": True, "bbox_original_coordinates": [[0, 0, 512, 1024]]}
        return {"views": {"original": copy.deepcopy(view), "hflip": copy.deepcopy(view)}}

    def cache_features(self, samples):
        feature_dir = self.output / "feature_cache"
        feature_dir.mkdir(exist_ok=True)
        for sample in samples:
            feature = torch.zeros(8, 4, 4)
            feature[0, :, :2], feature[1, :, 2:] = 1., 1.
            torch.save(feature, feature_dir / f"{sample.sample_id}.pt")

    def test_manual_label_firewall_and_category_free_manifest(self):
        label = next((self.root / "pv-gf2").rglob("labels/*.bmp"))
        with manual_label_firewall():
            with self.assertRaises(PermissionError):
                Image.open(label)
            first = prepare_rgb_manifest(self.config, self.output, self.logger)
        Image.fromarray(np.zeros((16, 16), np.uint8)).save(label)
        with manual_label_firewall():
            second = prepare_rgb_manifest(self.config, self.output, self.logger)
        self.assertEqual(first["samples"], second["samples"])
        for row in first["samples"]:
            self.assertEqual(set(row), {"sample_id", "image_path", "rgb_sha256", "split"})
            self.assertNotIn("正样本", row["image_path"])

    def test_ignored_pseudo_pixels_have_zero_gradient(self):
        logits = torch.randn(1, 2, 3, 3, requires_grad=True)
        target = torch.tensor([[[1, 0, 255], [255, 255, 255], [255, 255, 255]]])
        loss, _ = PartialPseudoLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[:, :, 1:].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[:, :, 0, 2].abs().sum()), 0.)
        changed = logits.detach().clone()
        changed[:, :, 1:] = 1000*torch.randn_like(changed[:, :, 1:])
        changed_loss, _ = PartialPseudoLoss()(changed, target)
        torch.testing.assert_close(changed_loss, loss.detach())

    def test_all_ignored_is_finite_zero(self):
        logits = torch.randn(2, 2, 4, 4, requires_grad=True)
        loss, _ = PartialPseudoLoss()(logits, torch.full((2, 4, 4), 255))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_flip_recovery_and_disagreement_abstention(self):
        box = [[600, 100, 900, 800]]
        self.assertEqual(restore_hflip_boxes(restore_hflip_boxes(box)), box)
        record = self.record()
        record["views"]["hflip"]["bbox_original_coordinates"] = [[512, 0, 1024, 1024]]
        _, _, accepted, iou, reason = view_agreement(record, self.settings)
        self.assertFalse(accepted.any())
        self.assertEqual(iou, 0.)
        self.assertEqual(reason, "view_disagreement")
        feature = np.zeros((2, 4, 4), np.float32)
        feature[0] = 1
        arrays, _ = build_pseudo_targets(feature, np.ones((4, 4), bool), np.zeros((4, 4), bool),
                                         np.eye(2, dtype=np.float32)[::-1], record, self.settings)
        self.assertTrue((arrays["target_filtered"] == 255).all())
        self.assertFalse(arrays["attention_filtered"].any())

    def test_prototypes_never_load_test_features(self):
        prepare_rgb_manifest(self.config, self.output, self.logger)
        samples = load_rgb_manifest(self.output / "rgb_manifest.json")
        self.cache_features([s for s in samples if s.split == "train"])
        _, _, used = fit_prototypes(samples, self.output, 4, 2)
        self.assertEqual(set(used), {s.sample_id for s in samples if s.split == "train"})

    def test_shuffling_is_split_local_derangement(self):
        samples = [RGBSample(str(i), "unused", str(i), "train" if i < 4 else "test") for i in range(6)]
        donors = shuffle_donors(samples, 42)
        splits = {s.sample_id: s.split for s in samples}
        self.assertEqual(set(donors), set(donors.values()))
        self.assertTrue(all(k != v and splits[k] == splits[v] for k, v in donors.items()))

    def test_nonobject_and_nan_teacher_json(self):
        self.assertIn("parse_warning", parse_teacher_response("[]"))
        self.assertEqual(parse_teacher_response('{"pv_exist":false,"confidence":NaN}')["confidence"], 0.)

    def test_training_without_annotation_files_then_sealed_final_evaluation(self):
        # Hide every annotation before the entire preparation/training pipeline.
        annotation_dirs = list((self.root / "pv-gf2").rglob("labels"))
        for folder in annotation_dirs:
            folder.rename(folder.with_name("held_out_gt"))
        with manual_label_firewall():
            prepare_rgb_manifest(self.config, self.output, self.logger)
            samples = load_rgb_manifest(self.output / "rgb_manifest.json")
            self.cache_features(samples)
            records = [{"sample_id": s.sample_id, **self.record()} for s in samples]
            atomic_write_json(self.output / "semantic_prompt.json", {"records": records})
            generate_pseudo_cache(self.config, self.output, self.logger)
            dataset = PseudoFeatureDataset(samples, self.output, "filtered", "filtered")
            self.assertEqual(set(dataset[0]), {"sample_id", "feature", "pseudo_target", "attention"})
            result = train_label_free(self.config, self.output, self.logger, device="cpu")
            self.assertEqual(len(result["runs"]), len(VARIANTS))
            seal_predictions(self.config, self.output)
            verify_seal(self.output)
            with self.assertRaisesRegex(RuntimeError, "sealed"):
                train_label_free(self.config, self.output, self.logger, device="cpu")
        for folder in annotation_dirs:
            folder.with_name("held_out_gt").rename(folder)
        final = evaluate_sealed(self.config, self.output, self.logger)
        self.assertFalse(final["manual_labels_used_for_training_or_selection"])
        self.assertEqual(final["test_images"], 2)
        path = next((self.output / "predictions").rglob("*.png"))
        with Image.open(path) as image:
            array = np.array(image)
        array[0, 0] = 255-array[0, 0]
        Image.fromarray(array).save(path)
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            verify_seal(self.output)


if __name__ == "__main__":
    unittest.main()
