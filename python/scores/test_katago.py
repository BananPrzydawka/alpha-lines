"""Nested bottleneck gradients and retired score-harness validation."""
import unittest

import torch
from torch import nn

from models.katago import NestedBottleneck
from scores.train import train
from config import settings


class KataGoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_nested_block_cost_and_gradients(self):
        block = NestedBottleneck(128, se_hidden=32, norm=settings["katago_model"]["group_norm"])
        convolution_weights = sum(
            layer.weight.numel() for layer in block.modules()
            if isinstance(layer, nn.Conv2d)
        )
        self.assertEqual(convolution_weights, 10 * 128 ** 2)
        features = torch.randn(2, 128, 10, 16, requires_grad=True)
        output = block(features)
        self.assertEqual(output.shape, features.shape)
        output.square().mean().backward()
        for name, parameter in block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_score_training_rejected(self):
        with self.assertRaisesRegex(ValueError, "only maia"):
            train("unused", "unused", model="katago", device="cpu")


if __name__ == '__main__':
    unittest.main()
