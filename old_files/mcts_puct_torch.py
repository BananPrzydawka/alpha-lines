import torch
from enum import Enum, auto
from config import board_size, num_parallel_games, mcts_num_simulations, mcts_c_puct, mcts_alpha, mcts_epsilon


class LeafOutcome(Enum):
    EXISTING_TERMINAL = auto()
    NEW_TERMINAL = auto()
    NEW_LEAF = auto()


class PuctMCTSNode:
    def __init__(self, game, prior_p0, prior_p1):
        self.game = game
        self.children = {}

        mask_p0, mask_p1, count_p0, count_p1 = game.get_legal_masks()
        self.legal_mask_p0 = mask_p0[0]
        self.legal_mask_p1 = mask_p1[0]
        self.num_legal_p0 = int(count_p0[0].item())
        self.num_legal_p1 = int(count_p1[0].item())

        p0_flat = prior_p0.flatten() * self.legal_mask_p0
        p1_flat = prior_p1.flatten() * self.legal_mask_p1
        self.prior_p0 = p0_flat / p0_flat.sum().clamp_min(1e-8)
        self.prior_p1 = p1_flat / p1_flat.sum().clamp_min(1e-8)

        self.visit_counts_p0 = torch.zeros(board_size, device=game.device)
        self.q_values_p0     = torch.zeros(board_size, device=game.device)
        self.visit_counts_p1 = torch.zeros(board_size, device=game.device)
        self.q_values_p1     = torch.zeros(board_size, device=game.device)

    def apply_dirichlet_noise(self):
        for prior, num_legal in [(self.prior_p0, self.num_legal_p0), (self.prior_p1, self.num_legal_p1)]:
            if num_legal > 0:
                noise = torch.distributions.Dirichlet(
                    torch.full((num_legal,), mcts_alpha, device=prior.device)
                ).sample()
                mask = prior > 0
                prior[mask] = (1 - mcts_epsilon) * prior[mask] + mcts_epsilon * noise
        # pass

    def select(self):
        total_p0 = self.visit_counts_p0.sum()
        total_p1 = self.visit_counts_p1.sum()
        scores_p0 = self.q_values_p0 + mcts_c_puct * self.prior_p0 * (torch.sqrt(total_p0) / (1 + self.visit_counts_p0))
        scores_p1 = self.q_values_p1 + mcts_c_puct * self.prior_p1 * (torch.sqrt(total_p1) / (1 + self.visit_counts_p1))
        scores_p0[self.legal_mask_p0 == 0] = -float('inf')
        scores_p1[self.legal_mask_p1 == 0] = -float('inf')
        return torch.argmax(scores_p0).item(), torch.argmax(scores_p1).item()


def _terminal_values(game):
    outcomes_p0, outcomes_p1 = game.get_terminal_outcomes()
    value_p0 = 1.0 if outcomes_p0[0] == 0 else (-1.0 if outcomes_p0[0] == 2 else 0.0)
    value_p1 = 1.0 if outcomes_p1[0] == 0 else (-1.0 if outcomes_p1[0] == 2 else 0.0)
    return value_p0, value_p1


class BatchedPuctMCTS:
    def search(self, game, model):
        encoded_all = torch.cat([game.get_encoded_states(0), game.get_encoded_states(1)], dim=0)
        priors, _, _, _ = model(encoded_all, apply_softmax=True)

        roots = []
        for i in range(num_parallel_games):
            root = PuctMCTSNode(game.clone_states_to_batch([i]), priors[i], priors[num_parallel_games + i])
            root.apply_dirichlet_noise()
            roots.append(root)

        for _ in range(mcts_num_simulations):
            self._simulate_all(roots, model)

        return [
            (
                root.visit_counts_p0 / root.visit_counts_p0.sum().clamp_min(1e-8),
                root.visit_counts_p1 / root.visit_counts_p1.sum().clamp_min(1e-8),
                root.q_values_p0.clone(),
                root.q_values_p1.clone(),
            )
            for root in roots
        ]

    def _find_leaf(self, root):
        node = root
        path = []  # list of (node, action_p0, action_p1)

        while True:
            if node.game.finished[0]:
                return path, node, LeafOutcome.EXISTING_TERMINAL

            action_p0, action_p1 = node.select()
            path.append((node, action_p0, action_p1))
            edge = (action_p0, action_p1)

            if edge not in node.children:
                child_game = node.game.clone_states_to_batch([0])
                child_game.action_step(
                    torch.tensor([action_p0], device=child_game.device),
                    torch.tensor([action_p1], device=child_game.device),
                )
                outcome = LeafOutcome.NEW_TERMINAL if child_game.finished[0] else LeafOutcome.NEW_LEAF
                return path, child_game, outcome

            node = node.children[edge]

    def _simulate_all(self, roots, model):
        leaf_data = [self._find_leaf(root) for root in roots]

        # Batch NN eval for all new non-terminal leaves
        new_leaf_indices = [i for i, (_, _, outcome) in enumerate(leaf_data) if outcome == LeafOutcome.NEW_LEAF]
        eval_results = {}

        if new_leaf_indices:
            M = len(new_leaf_indices)
            games = [leaf_data[i][1] for i in new_leaf_indices]
            encoded_all = torch.cat(
                [torch.cat([g.get_encoded_states(0) for g in games], dim=0),
                 torch.cat([g.get_encoded_states(1) for g in games], dim=0)],
                dim=0,
            )
            priors, value_logits, _, _ = model(encoded_all, apply_softmax=True)
            for batch_idx, sim_idx in enumerate(new_leaf_indices):
                eval_results[sim_idx] = (
                    priors[batch_idx], priors[M + batch_idx],
                    (value_logits[batch_idx, 0] - value_logits[batch_idx, 2]).item(),
                    (value_logits[M + batch_idx, 0] - value_logits[M + batch_idx, 2]).item(),
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
                        torch.zeros(board_size, device=node_or_game.device),
                        torch.zeros(board_size, device=node_or_game.device),
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