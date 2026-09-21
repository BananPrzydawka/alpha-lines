"""Maia score-harness integration, GAB gradients, and square-policy mapping."""
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from models.maia import MaiaNet
from scores.train import train


class MaiaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_learning_and_dynamic_bias(self):
        net = MaiaNet()
        inputs = torch.randn(2, 5, 10, 16)
        targets = torch.tensor([[3, 7], [12, 21]])
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        losses = []
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(inputs).flatten(0, 1), targets.flatten())
            losses.append(loss.item())
            loss.backward()
            for name, parameter in net.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(net.templates.grad.abs().sum().item(), 0)
            self.assertGreater(net.layers[0].gab[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertLess(losses[-1], losses[0])
        with torch.no_grad():
            tokens = net.embedding(inputs.flatten(2).index_select(2, net.indices).transpose(1, 2))
            weights = net.layers[0].gab(net.layers[0].norm1(tokens).mean(1))
            self.assertFalse(torch.allclose(weights[0], weights[1]))

    def test_scores_independent_of_batch_and_mode(self):
        net = MaiaNet()
        inputs = torch.randn(2, 5, 10, 16)
        with torch.no_grad():
            scores = net(inputs, apply_softmax=True)
            net.eval()
            single = net(inputs[:1], apply_softmax=True)
        self.assertEqual(scores.shape, (2, 2, 81))
        torch.testing.assert_close(scores[:1], single, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(scores.sum(-1), torch.ones(2, 2))

    def test_score_harness_checkpoint_and_compile(self):
        library = Path(__file__).resolve().parents[2] / 'rust/target/release/libalpha_lines_game.so'
        with tempfile.TemporaryDirectory() as output:
            result = train(library, output, model='maia', batch=2, steps=1,
                           validation_batch=2, validation_steps=1, device='cpu')
            checkpoint = torch.load(Path(output) / 'checkpoint.pt', weights_only=False)
            net = MaiaNet()
            net.load_state_dict(checkpoint['model'])
            self.assertEqual(result['options']['model'], 'maia')
            self.assertEqual(result['options']['model_batch'], 4)
            self.assertEqual(result['metrics'][-1]['step'], 1)
        inputs = torch.randn(2, 5, 10, 16)
        compiled = torch.compile(net, backend='aot_eager', fullgraph=True)
        torch.testing.assert_close(compiled(inputs), net(inputs))
        compiled(inputs).square().mean().backward()


if __name__ == '__main__':
    unittest.main()
