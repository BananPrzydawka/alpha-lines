"""Run after cargo build --release --lib: PYTHONPATH=python python -m unittest discover -s python -p test_scores.py."""
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from model import alpha_lines_net
from score_data import RandomScoreBatch, ScoreEncoder
from train_scores import train

LIBRARY = Path(__file__).resolve().parents[1] / "rust/target/release/libalpha_lines_game.so"


class ScoreDemoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_encoding_and_learning(self):
        encoder = ScoreEncoder()
        with RandomScoreBatch(LIBRARY, 4, 17) as batch:
            for _ in range(20):
                cells, targets = batch.step()
            inputs = encoder(cells)
            self.assertEqual(tuple(inputs.shape), (4, 5, 10, 16))
            self.assertTrue(torch.equal(inputs.sum(1), torch.ones(4, 10, 16)))
            self.assertTrue(torch.equal(inputs[:, 3].sum((1, 2)), (cells == 1).sum(1)))
            self.assertTrue(torch.equal(inputs[:, 4].sum((1, 2)), (cells == 2).sum(1)))
            saved_inputs = inputs.clone()
            labels = targets.clone()
            targets.fill_(80)
            self.assertTrue(torch.equal(encoder(cells), saved_inputs))
        net = alpha_lines_net(score_prediction=True)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        losses = []
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            logits = net(inputs)
            self.assertEqual(tuple(logits.shape), (4, 2, 81))
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            self.assertTrue(torch.isfinite(loss))
            losses.append(loss.item())
            loss.backward()
            self.assertGreater(net.conv_input.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertLess(losses[-1], losses[0])

    def test_expired_budget_still_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as output:
            result = train(LIBRARY, output, batch=2, steps=100, validation_steps=1,
                           eval_every=100, device="cpu", seconds=1e-9)
            self.assertEqual([r["step"] for r in result["metrics"]], [0])
            checkpoint = torch.load(Path(output) / "checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["step"], 0)

    def test_online_training_artifacts_and_default_model(self):
        with tempfile.TemporaryDirectory() as output:
            result = train(LIBRARY, output, batch=2, steps=2, validation_steps=2,
                           eval_every=2, device="cpu")
            self.assertEqual([r["step"] for r in result["metrics"]], [0, 2])
            checkpoint = torch.load(Path(output) / "checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["step"], 2)
            self.assertTrue((Path(output) / "metrics.csv").exists())
        net = alpha_lines_net().eval()
        with torch.no_grad():
            outputs = net(torch.zeros(1, 7, 10, 16))
        self.assertEqual([tuple(x.shape) for x in outputs], [(1, 10, 16), (1, 3), (1, 1), (1, 161)])


if __name__ == "__main__":
    unittest.main()
