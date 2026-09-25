"""Contracts for spatial action models and their shared score head."""
import copy
import unittest
from unittest.mock import patch

import torch

from config import settings
from models.katago import KataGoNet
from models.katago_tf import KataGoTFNet, SpatialTransformerBlock
from models.resnet import ResNet
from models.maia import MaiaNet
from models.factory import build_model


class ActionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_outputs_shared_gradients_and_compile(self):
        conv = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                    policy_filters=4,opponent_policy_filters=4,action_value_filters=4,
                    mark_class_filters=4,discounted_mark_filters=4,
                    immediate_score_filters=4,discounted_score_filters=4)
        configs = (
            ('katago',KataGoNet,conv),
            ('katago_tf',KataGoTFNet,dict(conv,attention_heads=2,mlp_ratio=2)),
            ('resnet',ResNet,conv),
            ('maia',MaiaNet,dict(dim=16,layers=1,head_dim=8,mlp_ratio=2,gab_dim=4,
                                  score_head=dict(square_features=4,hidden_dim=8))),
        )
        for name,model_class,options in configs:
            with self.subTest(model=model_class.__name__):
                net = model_class(options)
                self.assertIsInstance(build_model(name,options),model_class)
                board = torch.randn(2, 5, 10, 16)
                outputs = net(board)
                for i, output in enumerate(outputs):
                    self.assertEqual(output.shape, [(2, 10, 16), (2, 10, 16),
                                                    (2, 6, 10, 16), (2, 2, 81), (2, 2, 81),
                                                    (2, 8, 10, 16), (2, 10, 16)][i])
                    self.assertTrue(torch.isfinite(output).all())
                sum(x.square().mean() for x in outputs).backward()
                for name, parameter in net.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                net.zero_grad(set_to_none=True)
                net(board)[3].square().mean().backward()
                shared = net.layers if isinstance(net,MaiaNet) else net.tower
                self.assertGreater(next(shared.parameters()).grad.abs().sum(), 0)
                net.zero_grad(set_to_none=True)
                net(board)[4].square().mean().backward()
                self.assertGreater(next(shared.parameters()).grad.abs().sum(), 0)
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

    def test_transformer_replaces_both_inner_blocks(self):
        options = dict(settings['katago_tf_model'],filters=16,blocks=2,se_hidden=4,
                       attention_heads=2,mlp_ratio=2)
        net = KataGoTFNet(options)
        for outer in net.tower:
            self.assertEqual(len(outer.inner_blocks),2)
            self.assertTrue(all(isinstance(inner,SpatialTransformerBlock)
                                for inner in outer.inner_blocks))

    def test_maia_uses_only_playable_squares_and_zero_fills_other_outputs(self):
        options = dict(dim=16,layers=1,head_dim=8,mlp_ratio=2,gab_dim=4,
                       score_head=dict(square_features=4,hidden_dim=8))
        net = MaiaNet(options).eval()
        board = torch.randn(2,5,10,16)
        invalid = torch.ones(160,dtype=torch.bool)
        invalid[net.indices] = False
        changed = board.clone().flatten(2)
        changed[:,:,invalid] = torch.randn_like(changed[:,:,invalid])*100
        with torch.no_grad():
            original_outputs = net(board)
            changed_outputs = net(changed.reshape_as(board))
        for original, altered in zip(original_outputs, changed_outputs):
            torch.testing.assert_close(original,altered)
        for output in (original_outputs[i] for i in (0,1,2,5,6)):
            self.assertEqual(output.reshape(output.shape[0],-1,160)[:,:,invalid].count_nonzero().item(),0)

    def test_maia_big_version_uses_learned_gab_input_compression(self):
        options = dict(dim=16,layers=1,head_dim=8,mlp_ratio=2,gab_dim=8,
                       score_head=dict(square_features=4,hidden_dim=8))
        small = MaiaNet(dict(options,maia_big_version=False))
        big = MaiaNet(dict(options,maia_big_version=True))
        self.assertIsNone(small.layers[0].gab_input)
        self.assertEqual(small.layers[0].gab[0].in_features,16)
        self.assertEqual(big.layers[0].gab_input.out_features,32)
        self.assertEqual(big.layers[0].gab[0].in_features,80*32)
        board = torch.randn(2,5,10,16)
        outputs = big(board)
        self.assertEqual(outputs[0].shape,(2,10,16))
        sum(output.square().mean() for output in outputs).backward()
        self.assertGreater(big.layers[0].gab_input.weight.grad.abs().sum(),0)
        compiled = torch.compile(big.eval(),backend='aot_eager',fullgraph=True)
        for actual, expected in zip(compiled(board),outputs):
            torch.testing.assert_close(actual,expected)
        with self.assertRaisesRegex(ValueError,'maia_big_version'):
            MaiaNet(dict(options,maia_big_version=1))

    def test_configuration(self):
        baseline = copy.deepcopy(settings)
        options = copy.deepcopy(baseline['katago_model'])
        options.update(filters=16, blocks=1,
                       policy_filters=8, action_value_filters=4)
        options['group_norm']['groups'] = 2
        with patch.dict(settings, {'katago_model': options}):
            net = KataGoNet()
            self.assertEqual(net.norm_input.num_groups, 2)
            self.assertEqual(net.conv_input.out_channels, 16)
            self.assertEqual(len(net.tower), 1)


if __name__ == "__main__":
    unittest.main()
