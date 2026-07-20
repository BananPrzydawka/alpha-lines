import numpy as np
import torch
from enum import Enum, auto
from config import device, board_size, num_parallel_games, mcts_num_simulations, mcts_c_puct, mcts_alpha, mcts_epsilon


class LeafOutcome(Enum):
    EXISTING_TERMINAL = auto()
    NEW_TERMINAL = auto()
    NEW_LEAF = auto()


class PuctMCTSNode:
    def __init__(self, game, prior_p0, prior_p1):
        self.game = game
        self.children = {}

        mask_p0, mask_p1, count_p0, count_p1 = game.get_legal_masks()
        self.legal_mask_p0 = mask_p0[0] > 0
        self.legal_mask_p1 = mask_p1[0] > 0
        self.num_legal_p0 = int(count_p0[0])
        self.num_legal_p1 = int(count_p1[0])

        # prior_p0/prior_p1 are now plain numpy arrays by the time they reach
        # here (converted once, upstream, from the model's tensor output) —
        # no .cpu() call needed.
        p0_flat = np.asarray(prior_p0).flatten() * self.legal_mask_p0
        p1_flat = np.asarray(prior_p1).flatten() * self.legal_mask_p1
        self.prior_p0 = p0_flat / max(p0_flat.sum(), 1e-8)
        self.prior_p1 = p1_flat / max(p1_flat.sum(), 1e-8)

        self.visit_counts_p0 = np.zeros(board_size, dtype=np.float32)
        self.q_values_p0 = np.zeros(board_size, dtype=np.float32)
        self.visit_counts_p1 = np.zeros(board_size, dtype=np.float32)
        self.q_values_p1 = np.zeros(board_size, dtype=np.float32)

    def apply_dirichlet_noise(self):
        for prior, num_legal in [(self.prior_p0, self.num_legal_p0), (self.prior_p1, self.num_legal_p1)]:
            if num_legal > 0:
                noise = np.random.dirichlet(np.full(num_legal, mcts_alpha, dtype=np.float64)).astype(np.float32)
                mask = prior > 0
                prior[mask] = (1 - mcts_epsilon) * prior[mask] + mcts_epsilon * noise

    def select(self):
        total_p0 = self.visit_counts_p0.sum()
        total_p1 = self.visit_counts_p1.sum()
        scores_p0 = self.q_values_p0 + mcts_c_puct * self.prior_p0 * (np.sqrt(total_p0) / (1 + self.visit_counts_p0))
        scores_p1 = self.q_values_p1 + mcts_c_puct * self.prior_p1 * (np.sqrt(total_p1) / (1 + self.visit_counts_p1))
        scores_p0[~self.legal_mask_p0] = -np.inf
        scores_p1[~self.legal_mask_p1] = -np.inf
        return int(np.argmax(scores_p0)), int(np.argmax(scores_p1))


def _terminal_values(game):
    outcomes_p0, outcomes_p1 = game.get_terminal_outcomes()
    value_p0 = 1.0 if outcomes_p0[0] == 0 else (-1.0 if outcomes_p0[0] == 2 else 0.0)
    value_p1 = 1.0 if outcomes_p1[0] == 0 else (-1.0 if outcomes_p1[0] == 2 else 0.0)
    return value_p0, value_p1


def _encode_both_players(games):
    enc_p0 = np.concatenate([g.get_encoded_states(0) for g in games], axis=0)
    enc_p1 = np.concatenate([g.get_encoded_states(1) for g in games], axis=0)
    return torch.from_numpy(np.concatenate([enc_p0, enc_p1], axis=0)).to(device)


def _run_model(model, encoded_all):
    """Single choke point for model calls: no_grad (skip autograd bookkeeping
    we never use in MCTS) and one tensor->numpy conversion for priors/values
    instead of per-item .item()/indexing on the tensor."""
    with torch.no_grad():
        priors, value_logits, _, _ = model(encoded_all, apply_softmax=True)
    priors_np = priors.cpu().numpy()
    values_np = (value_logits[:, 0] - value_logits[:, 2]).cpu().numpy()
    return priors_np, values_np


class BatchedPuctMCTS:
    def search(self, game, model):
        encoded_all = _encode_both_players([game])
        priors_np, _ = _run_model(model, encoded_all)

        roots = []
        for i in range(num_parallel_games):
            root = PuctMCTSNode(
                game.clone_states_to_batch([i]),
                priors_np[i],
                priors_np[num_parallel_games + i],
            )
            root.apply_dirichlet_noise()
            roots.append(root)

        for _ in range(mcts_num_simulations):
            self._simulate_all(roots, model)

        return [
            (
                root.visit_counts_p0 / max(root.visit_counts_p0.sum(), 1e-8),
                root.visit_counts_p1 / max(root.visit_counts_p1.sum(), 1e-8),
                root.q_values_p0.copy(),
                root.q_values_p1.copy(),
            )
            for root in roots
        ]

    def _find_leaf(self, root):
        node = root
        path = []

        while True:
            if node.game.finished[0]:
                return path, node, LeafOutcome.EXISTING_TERMINAL

            action_p0, action_p1 = node.select()
            path.append((node, action_p0, action_p1))
            edge = (action_p0, action_p1)

            if edge not in node.children:
                child_game = node.game.clone_states_to_batch([0])
                child_game.action_step([action_p0], [action_p1])
                outcome = LeafOutcome.NEW_TERMINAL if child_game.finished[0] else LeafOutcome.NEW_LEAF
                return path, child_game, outcome

            node = node.children[edge]

    def _simulate_all(self, roots, model):
        leaf_data = [self._find_leaf(root) for root in roots]

        new_leaf_indices = [i for i, (_, _, outcome) in enumerate(leaf_data) if outcome == LeafOutcome.NEW_LEAF]
        eval_results = {}

        if new_leaf_indices:
            M = len(new_leaf_indices)
            games = [leaf_data[i][1] for i in new_leaf_indices]
            encoded_all = _encode_both_players(games)
            priors_np, values_np = _run_model(model, encoded_all)
            for batch_idx, sim_idx in enumerate(new_leaf_indices):
                eval_results[sim_idx] = (
                    priors_np[batch_idx], priors_np[M + batch_idx],
                    values_np[batch_idx], values_np[M + batch_idx],
                )

        for i, (path, node_or_game, outcome) in enumerate(leaf_data):
            match outcome:
                case LeafOutcome.EXISTING_TERMINAL:
                    value_p0, value_p1 = _terminal_values(node_or_game.game)

                case LeafOutcome.NEW_TERMINAL:
                    value_p0, value_p1 = _terminal_values(node_or_game)
                    parent_node, a0, a1 = path[-1]
                    parent_node.children[(a0, a1)] = PuctMCTSNode(
                        node_or_game,
                        np.zeros(board_size, dtype=np.float32),
                        np.zeros(board_size, dtype=np.float32),
                    )

                case LeafOutcome.NEW_LEAF:
                    prior_p0, prior_p1, value_p0, value_p1 = eval_results[i]
                    parent_node, a0, a1 = path[-1]
                    parent_node.children[(a0, a1)] = PuctMCTSNode(node_or_game, prior_p0, prior_p1)

            for node, a0, a1 in reversed(path):
                node.q_values_p0[a0] = (node.q_values_p0[a0] * node.visit_counts_p0[a0] + value_p0) / (node.visit_counts_p0[a0] + 1)
                node.visit_counts_p0[a0] += 1
                node.q_values_p1[a1] = (node.q_values_p1[a1] * node.visit_counts_p1[a1] + value_p1) / (node.visit_counts_p1[a1] + 1)
                node.visit_counts_p1[a1] += 1