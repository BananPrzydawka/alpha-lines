"""Run after cargo build --release --lib: PYTHONPATH=python python -m unittest discover -s python/scores -p test_scores.py."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.maia import MaiaNet
from scores.data import RandomScoreBatch, ScoreEncoder, both_perspectives
from scores.train import train, format_metrics

LIBRARY = Path(__file__).resolve().parents[2] / "rust/target/release/libalpha_lines_game.so"


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
        net = MaiaNet()
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
            self.assertGreater(net.embedding.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertLess(losses[-1], losses[0])

    def test_normalization_independent_of_batch_and_mode(self):
        net = MaiaNet()
        inputs = torch.randn(3, 5, 10, 16)
        with torch.no_grad():
            training = net(inputs)
            net.eval()
            evaluation = net(inputs)
            alone = net(inputs[:1])
        torch.testing.assert_close(training, evaluation)
        torch.testing.assert_close(evaluation[:1], alone, atol=1e-5, rtol=1e-4)

    def test_perspective_swap(self):
        cells = torch.arange(80, dtype=torch.uint8).remainder(4).unsqueeze(0)
        swapped_cells = cells.clone()
        swapped_cells[cells == 1] = 2
        swapped_cells[cells == 2] = 1
        encoder = ScoreEncoder()
        inputs = encoder(cells)
        targets = torch.tensor([[7, 23]])
        augmented, labels = both_perspectives(inputs, targets)
        self.assertTrue(torch.equal(augmented[:1], inputs))
        self.assertTrue(torch.equal(augmented[1:], encoder(swapped_cells)))
        self.assertEqual(labels.tolist(), [[7, 23], [23, 7]])
        self.assertTrue(augmented.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(targets.tolist(), [[7, 23]])

    def test_expired_budget_still_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as output:
            result = train(LIBRARY, output, batch=2, steps=100, validation_steps=1,
                           eval_every=100, device="cpu", validation_batch=2, seconds=1e-9)
            self.assertEqual([r["step"] for r in result["metrics"]], [0])
            checkpoint = torch.load(Path(output) / "checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["step"], 0)

    def test_artifacts_saved_once_after_final_report(self):
        with tempfile.TemporaryDirectory() as output:
            output = Path(output)
            reports = []
            commits = []
            original_format = format_metrics

            def check_report(row):
                self.assertEqual(list(output.iterdir()), [])
                reports.append(row["step"])
                return original_format(row)

            def commit():
                commits.append(reports.copy())
                checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)
                self.assertEqual(checkpoint["step"], 3)
                self.assertTrue(checkpoint["optimizer"]["state"])
                saved = json.loads((output / "results.json").read_text())
                self.assertEqual([row["step"] for row in saved["metrics"]], reports)
                with (output / "metrics.csv").open(newline="") as stream:
                    self.assertEqual([int(row["step"]) for row in csv.DictReader(stream)], reports)

            with patch("scores.train.format_metrics", side_effect=check_report), \
                    patch("scores.train.torch.save", wraps=torch.save) as save:
                train(LIBRARY, output, batch=2, steps=3, validation_steps=1,
                      eval_every=1, device="cpu", validation_batch=2, on_save=commit)
                self.assertEqual(save.call_count, 1)
            self.assertEqual(commits, [[0, 1, 2, 3]])

    def test_online_training_artifacts_and_default_model(self):
        with tempfile.TemporaryDirectory() as output:
            result = train(LIBRARY, output, batch=2, steps=2, validation_steps=2,
                           eval_every=2, device="cpu", validation_batch=2)
            self.assertEqual([r["step"] for r in result["metrics"]], [0, 2])
            checkpoint = torch.load(Path(output) / "checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(checkpoint["options"]["model_batch"], 4)
            self.assertEqual(result["metrics"][-1]["examples"], 8)
            first, last = result["metrics"]
            for field in ("cpu_s", "model_s", "logging_s"):
                self.assertGreater(last[field], first[field])
            self.assertLessEqual(sum(last[key] for key in ("cpu_s", "model_s", "logging_s")),
                                 last["elapsed_s"])
            line = format_metrics(last)
            self.assertNotIn("MAE", line)
            self.assertIn("cpu ", line)
            self.assertIn("model ", line)
            self.assertIn("log ", line)
            self.assertIn("ms", line)
            self.assertEqual(first["cpu_ms"], 0)
            self.assertAlmostEqual(first["log_ms"], 1000 * first["logging_s"])
            self.assertAlmostEqual(last["model_ms"],
                                   1000 * (last["model_s"] - first["model_s"]))
            self.assertTrue((Path(output) / "metrics.csv").exists())



if __name__ == "__main__":
    unittest.main()
