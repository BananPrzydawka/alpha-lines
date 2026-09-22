"""Contracts for spatial action models and their shared score head."""
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

    def test_outputs_shared_gradients_and_compile(self):
        for model_class in (ResNet, KataGoNet):
            with self.subTest(model=model_class.__name__):
                net = model_class()
                board = torch.randn(2, 5, 10, 16)
                outputs = net(board)
                for i, output in enumerate(outputs):
                    self.assertEqual(output.shape, [(2, 10, 16), (2, 10, 16),
                                                    (2, 6, 10, 16), (2, 2, 81), (2, 2, 81)][i])
                    self.assertTrue(torch.isfinite(output).all())
                sum(x.square().mean() for x in outputs).backward()
                for name, parameter in net.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                net.zero_grad(set_to_none=True)
                net(board)[3].square().mean().backward()
                self.assertGreater(next(net.tower.parameters()).grad.abs().sum(), 0)
                net.zero_grad(set_to_none=True)
                net(board)[4].square().mean().backward()
                self.assertGreater(next(net.tower.parameters()).grad.abs().sum(), 0)
                net.eval()
                alone = net(board[0])
                for full, single in zip(outputs, alone):
                    torch.testing.assert_close(full[:1], single, atol=1e-5, rtol=1e-4)
                compiled = torch.compile(net, backend="aot_eager", fullgraph=True)
                for actual, expected in zip(compiled(board), outputs):
                    torch.testing.assert_close(actual, expected)
                bf16 = net.to(torch.bfloat16)(board.to(torch.bfloat16))
                self.assertTrue(all(x.dtype == torch.bfloat16 for x in bf16))
                with self.assertRaisesRegex(ValueError, "board must"):
                    net(board[:, :4])

    def test_independent_configuration(self):
        baseline = copy.deepcopy(settings)
        for key, model_class, other_class in (
            ("resnet_model", ResNet, KataGoNet),
            ("katago_model", KataGoNet, ResNet),
        ):
            original_other = sum(p.numel() for p in other_class().parameters())
            options = copy.deepcopy(baseline[key])
            options.update(filters=16, blocks=1,
                           policy_filters=8, action_value_filters=4)
            options['group_norm']['groups'] = 2
            with patch.dict(settings, {key: options}):
                net = model_class()
                self.assertEqual(net.norm_input.num_groups, 2)
                self.assertEqual(net.conv_input.out_channels, 16)
                self.assertEqual(sum(p.numel() for p in other_class().parameters()), original_other)


if __name__ == "__main__":
    unittest.main()
