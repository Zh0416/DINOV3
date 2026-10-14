import unittest

import torch
from torch import nn

from pvseg.losses import DiceCrossEntropyLoss
from pvseg.metrics import SegmentationMetrics
from pvseg.models import CachedFeatureSegmenter, FrozenDINOv3


class _Output:
    def __init__(self, tensor):
        self.last_hidden_state = tensor


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        return _Output(torch.arange(batch * 5 * 8, dtype=pixel_values.dtype).reshape(batch, 5, 8))


class _TimmBackbone(_Backbone):
    def forward_features(self, images):
        return super().forward(images).last_hidden_state


class ModelAndMetricTests(unittest.TestCase):
    def test_frozen_dino_patch_reshape(self):
        model = FrozenDINOv3(_Backbone(), [0, 0, 0], [1, 1, 1], 32, 16, 8)
        result = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(result.shape), (2, 8, 2, 2))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_timm_dino_patch_reshape(self):
        model = FrozenDINOv3(
            _TimmBackbone(), [0, 0, 0], [1, 1, 1], 32, 16, 8, backend="timm"
        )
        result = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(result.shape), (2, 8, 2, 2))

    def test_adapter_and_decoder_shapes(self):
        model = CachedFeatureSegmenter(16, 32, 8, use_adapter=True)
        logits = model(torch.randn(2, 16, 8, 8), torch.ones(2, 8, 8), (64, 64))
        self.assertEqual(tuple(logits.shape), (2, 2, 64, 64))
        loss, parts = DiceCrossEntropyLoss()(logits, torch.zeros(2, 64, 64, dtype=torch.long))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("dice_loss", parts)

    def test_perfect_metrics(self):
        target = torch.tensor([[[0, 1], [0, 1]]])
        logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]], [[-10.0, 10.0], [-10.0, 10.0]]]])
        meter = SegmentationMetrics()
        meter.update(logits, target, ["困难正样本"])
        metrics = meter.compute()["overall"]
        self.assertEqual(metrics["miou"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

