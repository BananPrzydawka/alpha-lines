class Exp3MCTS:
    def __init__(self, num_sims=None):
        self.num_sims = num_sims if num_sims is not None else mcts_num_simulations

    def _evaluate(self, game, model):
        encoded_states = torch.cat(
            [game.get_encoded_states(0), game.get_encoded_states(1)], dim=0
        )
        priors, value_logits, _, _ = model(encoded_states, apply_softmax=True)
        prior_p0 = priors[0]
        prior_p1 = priors[1]
        value_p0 = (value_logits[0, 0] - value_logits[0, 2]).item()
        value_p1 = (value_logits[1, 0] - value_logits[1, 2]).item()
        return prior_p0, prior_p1, value_p0, value_p1

    def search(self, game, model=None):
        root_game = game.clone_states_to_batch([0])

        if model is not None:
            prior_p0, prior_p1, _, _ = self._evaluate(root_game, model)
            root = Exp3MCTSNode(root_game, prior_p0=prior_p0, prior_p1=prior_p1)
        else:
            root = Exp3MCTSNode(root_game)

        for _ in range(self.num_sims):
            self._simulate(root, model)

        policy_p0 = root.strategy_sum_p0 / root.strategy_sum_p0.sum().clamp_min(1e-8)
        policy_p1 = root.strategy_sum_p1 / root.strategy_sum_p1.sum().clamp_min(1e-8)
        return policy_p0, policy_p1, None, None

    def _simulate(self, node, model):
        if node.game.finished[0]:
            outcomes_p0, outcomes_p1 = node.game.get_terminal_outcomes()
            value_p0 = 1.0 if outcomes_p0[0] == 0 else (-1.0 if outcomes_p0[0] == 2 else 0.0)
            value_p1 = 1.0 if outcomes_p1[0] == 0 else (-1.0 if outcomes_p1[0] == 2 else 0.0)
            return value_p0, value_p1

        action_p0, action_p1, prob_p0, prob_p1 = node.select()
        edge = (action_p0, action_p1)

        if edge in node.children:
            value_p0, value_p1 = self._simulate(node.children[edge], model)
        else:
            child_game = node.game.clone_states_to_batch([0])
            child_game.action_step(
                torch.tensor([action_p0], device=child_game.device),
                torch.tensor([action_p1], device=child_game.device),
            )

            if child_game.finished[0]:
                outcomes_p0, outcomes_p1 = child_game.get_terminal_outcomes()
                value_p0 = 1.0 if outcomes_p0[0] == 0 else (-1.0 if outcomes_p0[0] == 2 else 0.0)
                value_p1 = 1.0 if outcomes_p1[0] == 0 else (-1.0 if outcomes_p1[0] == 2 else 0.0)
                node.children[edge] = Exp3MCTSNode(child_game)
            elif model is not None:
                prior_p0, prior_p1, value_p0, value_p1 = self._evaluate(child_game, model)
                node.children[edge] = Exp3MCTSNode(child_game, prior_p0=prior_p0, prior_p1=prior_p1)
            else:
                value_p0, value_p1 = 0.0, 0.0
                node.children[edge] = Exp3MCTSNode(child_game)

        node.update_weights(action_p0, action_p1, value_p0, value_p1, prob_p0, prob_p1)
        return value_p0, value_p1