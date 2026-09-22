"""Contracts and score conditioning for spatial action models."""
import copy
import unittest
from unittest.mock import patch

import torch

from config import settings
from models.resnet import ResNet
from models.katago import KataGoNet


class ActionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_outputs_conditioning_gradients_and_compile(self):
        for model_class in (ResNet, KataGoNet):
            with self.subTest(model=model_class.__name__):
                net = model_class()
                board = torch.randn(2, 5, 10, 16)
                scores = torch.tensor([[0, 80], [41, 40]])
                outputs = net(board, scores)
                changed = net(board, scores.flip(1))
                for i, (output, other) in enumerate(zip(outputs, changed)):
                    self.assertEqual(output.shape, (2, 6, 10, 16) if i == 2 else (2, 10, 16))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertFalse(torch.allclose(output, other))
                sum(x.square().mean() for x in outputs).backward()
                for name, parameter in net.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(net.score_embed[0].weight.grad.abs().sum(), 0)
                net.eval()
                alone = net(board[0], scores[0])
                for full, single in zip(outputs, alone):
                    torch.testing.assert_close(full[:1], single, atol=1e-5, rtol=1e-4)
                compiled = torch.compile(net, backend="aot_eager", fullgraph=True)
                for actual, expected in zip(compiled(board, scores), outputs):
                    torch.testing.assert_close(actual, expected)
                bf16 = net.to(torch.bfloat16)(board.to(torch.bfloat16), scores)
                self.assertTrue(all(x.dtype == torch.bfloat16 for x in bf16))
                with self.assertRaisesRegex(ValueError, "scores must"):
                    net(board, scores[:1])

    def test_independent_configuration(self):
        baseline = copy.deepcopy(settings)
        for key, model_class, other_class in (
            ("resnet_model", ResNet, KataGoNet),
            ("katago_model", KataGoNet, ResNet),
        ):
            original_other = sum(p.numel() for p in other_class().parameters())
            options = copy.deepcopy(baseline[key])
            options.update(filters=16, blocks=1, score_embed_hidden=7,
                           policy_filters=8, action_value_filters=4)
            options['group_norm']['groups'] = 2
            with patch.dict(settings, {key: options}):
                net = model_class()
                self.assertEqual(net.norm_input.num_groups, 2)
                self.assertEqual(net.conv_input.out_channels, 16)
                self.assertEqual(sum(p.numel() for p in other_class().parameters()), original_other)


if __name__ == "__main__":
    unittest.main()
