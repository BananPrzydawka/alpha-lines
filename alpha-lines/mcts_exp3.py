import numpy as np
import torch
from enum import Enum, auto
from config import device, board_size, mcts_num_simulations, mcts_epx3_gamma, num_parallel_games


class LeafOutcome(Enum):
    EXISTING_TERMINAL = auto()
    NEW_TERMINAL = auto()
    NEW_LEAF = auto()


def _sample(probs):
    """Categorical sample via inverse-CDF (cumsum + searchsorted), avoiding
    np.random.choice's per-call argument-validation overhead on this hot path."""
    cum = np.cumsum(probs)
    idx = np.searchsorted(cum, np.random.random() * cum[-1], side='right')
    return min(int(idx), len(probs) - 1)


class Exp3MCTSNode:
    """Numpy-backed: legal masks, log-weights, and strategy sums are all plain arrays.
    torch only appears at construction, to accept a prior straight from the model
    (converted once via np.asarray) — everything downstream (select/update_weights,
    called many times per node) runs in numpy, avoiding torch's per-op dispatch
    overhead on these 160-element tensors."""

    def __init__(self, game, prior_p0=None, prior_p1=None):
        self.game = game
        self.children = {}

        mask_p0, mask_p1, count_p0, count_p1 = game.get_legal_masks()
        self.legal_mask_p0 = mask_p0[0] > 0
        self.legal_mask_p1 = mask_p1[0] > 0
        self.num_legal_p0 = int(count_p0[0])
        self.num_legal_p1 = int(count_p1[0])

        self.log_weights_p0 = self._init_log_weights(prior_p0, self.legal_mask_p0)
        self.log_weights_p1 = self._init_log_weights(prior_p1, self.legal_mask_p1)

        self.strategy_sum_p0 = np.zeros(board_size, dtype=np.float32)
        self.strategy_sum_p1 = np.zeros(board_size, dtype=np.float32)

    @staticmethod
    def _init_log_weights(prior, legal_mask):
        log_weights = np.full(legal_mask.shape[0], -np.inf, dtype=np.float32)
        if prior is not None:
            p = np.asarray(prior).flatten()
            log_weights[legal_mask] = np.log(np.clip(p[legal_mask], 1e-8, None))
        else:
            log_weights[legal_mask] = 0.0
        return log_weights

    def _mixed_strategy(self, log_weights, legal_mask, num_legal, gamma):
        legal_weights = log_weights[legal_mask]
        exp_w = np.exp(legal_weights - legal_weights.max())
        weights = exp_w / exp_w.sum()
        mixed = np.zeros_like(log_weights)
        mixed[legal_mask] = (1 - gamma) * weights + gamma / num_legal
        return mixed

    def select(self):
        strategy_p0 = self._mixed_strategy(self.log_weights_p0, self.legal_mask_p0, self.num_legal_p0, mcts_epx3_gamma)
        strategy_p1 = self._mixed_strategy(self.log_weights_p1, self.legal_mask_p1, self.num_legal_p1, mcts_epx3_gamma)
        self.strategy_sum_p0 += strategy_p0
        self.strategy_sum_p1 += strategy_p1
        action_p0 = _sample(strategy_p0)
        action_p1 = _sample(strategy_p1)
        return action_p0, action_p1, float(strategy_p0[action_p0]), float(strategy_p1[action_p1])

    def update_weights(self, action_p0, action_p1, value_p0, value_p1, prob_p0, prob_p1):
        reward_p0 = (value_p0 + 1) / 2
        reward_p1 = (value_p1 + 1) / 2
        self.log_weights_p0[action_p0] += mcts_epx3_gamma * (reward_p0 / prob_p0) / self.num_legal_p0
        self.log_weights_p1[action_p1] += mcts_epx3_gamma * (reward_p1 / prob_p1) / self.num_legal_p1


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
    with torch.no_grad():
        priors, value_logits, _, _ = model(encoded_all, apply_softmax=True)
    priors_np = priors.cpu().numpy()
    values_np = (value_logits[:, 0] - value_logits[:, 2]).cpu().numpy()
    return priors_np, values_np


class BatchedExp3MCTS:
    def __init__(self, num_sims=None):
        self.num_sims = num_sims if num_sims is not None else mcts_num_simulations

    def search(self, game, model):
        encoded_all = _encode_both_players([game])
        priors_np, _ = _run_model(model, encoded_all)


        roots = [
            Exp3MCTSNode(
                game.clone_states_to_batch([i]),
                prior_p0=priors_np[i],
                prior_p1=priors_np[num_parallel_games + i],
            )
            for i in range(num_parallel_games)
        ]


        for _ in range(self.num_sims):
            self._simulate_all(roots, model)

        return [
            (
                root.strategy_sum_p0 / max(root.strategy_sum_p0.sum(), 1e-8),
                root.strategy_sum_p1 / max(root.strategy_sum_p1.sum(), 1e-8),
                None,
                None,
            )
            for root in roots
        ]

    def _find_leaf(self, root):
        node = root
        path = []  # list of (node, action_p0, action_p1, prob_p0, prob_p1)

        while True:
            if node.game.finished[0]:
                return path, node, LeafOutcome.EXISTING_TERMINAL

            action_p0, action_p1, prob_p0, prob_p1 = node.select()
            path.append((node, action_p0, action_p1, prob_p0, prob_p1))
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
                    parent_node, a0, a1, _, _ = path[-1]
                    parent_node.children[(a0, a1)] = Exp3MCTSNode(node_or_game)

                case LeafOutcome.NEW_LEAF:
                    prior_p0, prior_p1, value_p0, value_p1 = eval_results[i]
                    parent_node, a0, a1, _, _ = path[-1]
                    parent_node.children[(a0, a1)] = Exp3MCTSNode(node_or_game, prior_p0=prior_p0, prior_p1=prior_p1)

            for node, a0, a1, prob_p0, prob_p1 in reversed(path):
                node.update_weights(a0, a1, value_p0, value_p1, prob_p0, prob_p1)