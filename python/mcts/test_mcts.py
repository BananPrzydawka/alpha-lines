import unittest

import torch

from mcts.train import default_node_capacity, losses, transforms_for


class MctsTrainingTests(unittest.TestCase):
    def test_historical_node_capacity_tracks_games_and_simulations(self):
        self.assertEqual(default_node_capacity(2048, 100), 1638400)
        self.assertEqual(default_node_capacity(4096, 100), 3276800)
        self.assertEqual(default_node_capacity(2048, 100, 4), 819200)

    def test_value_loss_ignores_unvisited_moves_and_can_weight_by_visits(self):
        logits = torch.zeros((1, 10, 16))
        q = torch.zeros((1, 10, 16), requires_grad=True)
        with torch.no_grad():
            q.flatten()[0] = 1
            q.flatten()[1] = 3
            q.flatten()[2] = 100
        policy = torch.zeros((1, 160))
        policy[0, 0] = 1
        target = torch.zeros((1, 160))
        visits = torch.zeros((1, 160))
        visits[0, 0] = 1
        visits[0, 1] = 3
        _, equal = losses(logits, q, policy, target, visits, 'equal')
        _, weighted = losses(logits, q, policy, target, visits, 'visits')
        self.assertAlmostEqual(equal.item(), 5)
        self.assertAlmostEqual(weighted.item(), 7)
        equal.backward()
        self.assertEqual(q.grad.flatten()[2].item(), 0)

    def test_double_augmentation_is_identity_plus_180_per_position(self):
        ids, transforms = transforms_for(7, double_flip_augment=True)
        for position in range(7):
            self.assertEqual(sorted(transforms[ids == position].tolist()), [0, 3])

    def test_all_augmentation_takes_precedence_over_double(self):
        ids, transforms = transforms_for(7, double_flip_augment=True, all_flip_augment=True)
        for position in range(7):
            self.assertEqual(sorted(transforms[ids == position].tolist()), [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
