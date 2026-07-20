"""
Monte Carlo Tree Search for Alpha-Lines with simultaneous moves
"""

import numpy as np
import torch
import math
from config import (
    MCTS_C_PUCT, MCTS_DIRICHLET_ALPHA, MCTS_DIRICHLET_WEIGHT,
    BOARD_HEIGHT, BOARD_WIDTH
)


class MCTSNode:
    """
    A node in the MCTS tree for simultaneous move games.
    Each child represents a pair of moves (move_x, move_o).
    """
    
    def __init__(self, game, player, prior_prob=0.0):
        """
        game: LinesGame instance
        player: which player this node is from the perspective of (1 for X, -1 for O)
        prior_prob: prior probability from the policy network
        """
        self.game = game  # LinesGame instance
        self.player = player  # Perspective player for this node
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # (move_x, move_o) -> MCTSNode
        
        # Q value
        self.q_value = 0.0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_terminal(self):
        return self.game.game_over
    
    def expand(self, policy_x, policy_o):
        """
        Expand the node by creating children for all pairs of legal moves.
        policy_x: 10x16 numpy array of probabilities for X
        policy_o: 10x16 numpy array of probabilities for O
        """
        legal_x = self.game.get_legal_moves(1)  # X's legal moves
        legal_o = self.game.get_legal_moves(-1)  # O's legal moves
        
        # Handle case where one player has no legal moves
        if not legal_x and not legal_o:
            return
        
        if not legal_x:
            # Only O moves
            for move_o in legal_o:
                child_game = self.game.clone()
                child_game.execute_moves(None, move_o)
                prior = policy_o[move_o]
                self.children[(None, move_o)] = MCTSNode(child_game, self.player, prior_prob=prior)
            return
        
        if not legal_o:
            # Only X moves
            for move_x in legal_x:
                child_game = self.game.clone()
                child_game.execute_moves(move_x, None)
                prior = policy_x[move_x]
                self.children[(move_x, None)] = MCTSNode(child_game, self.player, prior_prob=prior)
            return
        
        # Both players have legal moves - create children for all pairs
        for move_x in legal_x:
            for move_o in legal_o:
                child_game = self.game.clone()
                child_game.execute_moves(move_x, move_o)
                # Prior is product of individual policy probabilities
                prior = policy_x[move_x] * policy_o[move_o]
                self.children[(move_x, move_o)] = MCTSNode(child_game, self.player, prior_prob=prior)
    
    def select_child(self, c_puct=MCTS_C_PUCT):
        """
        Select the child with the highest UCB score.
        """
        best_score = -float('inf')
        best_child = None
        
        for move_pair, child in self.children.items():
            ucb = self._ucb_score(child, c_puct)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        
        return best_child
    
    def _ucb_score(self, child, c_puct):
        """
        Calculate UCB score with PUCT formula.
        """
        if child.visit_count == 0:
            q_score = 0.0
        else:
            q_score = child.q_value
        
        u_score = c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
        
        return q_score + u_score
    
    def backup(self, value):
        """
        Backup the value through the tree.
        value is from the perspective of the root player.
        """
        self.visit_count += 1
        self.value_sum += value
        self.q_value = self.value_sum / self.visit_count
    
    def get_action_distribution(self, temperature=1.0):
        """
        Get the action distribution based on visit counts.
        Returns distributions for both players.
        """
        # Aggregate visit counts for each player's moves
        visit_counts_x = {}
        visit_counts_o = {}
        
        for (move_x, move_o), child in self.children.items():
            if move_x is not None:
                visit_counts_x[move_x] = visit_counts_x.get(move_x, 0) + child.visit_count
            if move_o is not None:
                visit_counts_o[move_o] = visit_counts_o.get(move_o, 0) + child.visit_count
        
        def get_probs(visit_counts, legal_moves):
            if not legal_moves:
                return None, None
            counts = np.array([visit_counts.get(m, 0) for m in legal_moves], dtype=np.float32)
            
            if temperature == 0 or counts.sum() == 0:
                best_idx = np.argmax(counts)
                probs = np.zeros_like(counts)
                probs[best_idx] = 1.0
            else:
                counts = counts ** (1.0 / temperature)
                total = counts.sum()
                if total > 0:
                    probs = counts / total
                else:
                    probs = np.ones_like(counts) / len(counts)
            
            return legal_moves, probs
        
        legal_x = self.game.get_legal_moves(1)
        legal_o = self.game.get_legal_moves(-1)
        
        moves_x, probs_x = get_probs(visit_counts_x, legal_x)
        moves_o, probs_o = get_probs(visit_counts_o, legal_o)
        
        return moves_x, probs_x, moves_o, probs_o


class MCTS:
    """
    Monte Carlo Tree Search for simultaneous move games.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def search(self, game, player, num_simulations, add_dirichlet_noise=True):
        """
        Run MCTS from the given game state for a specific player.
        Returns the root node after search.
        """
        root = MCTSNode(game, player)
        
        # Evaluate the root position
        policy_x, policy_o, value = self._evaluate(game, player)
        root.expand(policy_x, policy_o)
        
        return self._search_with_path(root, num_simulations, add_dirichlet_noise)
    
    def _search_with_path(self, root, num_simulations, add_dirichlet_noise=True):
        """
        Run MCTS with proper path tracking for backup.
        """
        if add_dirichlet_noise and not root.game.game_over:
            self._add_dirichlet_noise(root)
        
        for _ in range(num_simulations):
            path = []
            node = root
            
            # Select - follow the tree policy
            while not node.is_leaf() and not node.is_terminal():
                path.append(node)
                node = node.select_child()
            
            # Check if terminal
            if node.is_terminal():
                # Game is over, use actual outcome from root player's perspective
                if node.game.winner == root.player:
                    value = 1.0
                elif node.game.winner == -root.player:
                    value = -1.0
                else:
                    value = 0.0
            else:
                # Expand and evaluate
                policy_x, policy_o, value = self._evaluate(node.game, root.player)
                node.expand(policy_x, policy_o)
                path.append(node)
            
            # Backup - value is from the perspective of the root player
            # All nodes in the tree store values from the root player's perspective
            for n in reversed(path):
                n.backup(value)
        
        return root
    
    def _evaluate(self, game, root_player):
        """
        Evaluate a game state using the neural network.
        Returns: policy_x (10x16 numpy), policy_o (10x16 numpy), value (scalar)
        The value is from the perspective of root_player.
        """
        # Get policy and value from X's perspective
        state_x = game.get_state_encoding(perspective=1)
        state_tensor = torch.from_numpy(state_x).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value, point_diff = self.model(state_tensor)
        
        policy = policy.squeeze(0).cpu().numpy()
        value = value.squeeze(0).cpu().item()
        
        # Value from network is from the perspective of the player to move
        # But in simultaneous games, both players "move" at once
        # The network encodes from perspective=1 (X), so value is from X's perspective
        # We need to convert to root_player's perspective
        if root_player == -1:
            value = -value
        
        # Mask illegal moves for each player
        legal_mask_x = game.get_legal_moves_mask(1)
        legal_mask_o = game.get_legal_moves_mask(-1)
        
        policy_x = policy * legal_mask_x
        policy_o = policy * legal_mask_o  # Same policy, but masked for O's legal moves
        
        # Renormalize
        policy_sum_x = policy_x.sum()
        if policy_sum_x > 0:
            policy_x = policy_x / policy_sum_x
        else:
            policy_x = legal_mask_x / legal_mask_x.sum() if legal_mask_x.sum() > 0 else legal_mask_x
        
        policy_sum_o = policy_o.sum()
        if policy_sum_o > 0:
            policy_o = policy_o / policy_sum_o
        else:
            policy_o = legal_mask_o / legal_mask_o.sum() if legal_mask_o.sum() > 0 else legal_mask_o
        
        return policy_x, policy_o, value
    
    def _add_dirichlet_noise(self, node):
        """
        Add Dirichlet noise to the prior probabilities of the root node's children.
        """
        if not node.children:
            return
        
        legal_moves = list(node.children.keys())
        noise = np.random.dirichlet([MCTS_DIRICHLET_ALPHA] * len(legal_moves))
        
        for i, move_pair in enumerate(legal_moves):
            node.children[move_pair].prior_prob = (
                (1 - MCTS_DIRICHLET_WEIGHT) * node.children[move_pair].prior_prob +
                MCTS_DIRICHLET_WEIGHT * noise[i]
            )
    
    def get_actions(self, game, num_simulations, temperature=1.0, sample=True):
        """
        Get the best actions for both players from MCTS.
        If sample=True, sample from the visit count distribution.
        If sample=False, pick the most visited moves (deterministic).
        """
        # Run MCTS from X's perspective (arbitrary, since it's symmetric)
        root = self.search(game, player=1, num_simulations=num_simulations, 
                          add_dirichlet_noise=(temperature > 0))
        
        if root.game.game_over:
            return None, None, None
        
        moves_x, probs_x, moves_o, probs_o = root.get_action_distribution(temperature=temperature)
        
        move_x = None
        move_o = None
        
        if moves_x is not None and probs_x is not None:
            if sample and temperature > 0:
                idx = np.random.choice(len(moves_x), p=probs_x)
            else:
                idx = np.argmax(probs_x)
            move_x = moves_x[idx]
        
        if moves_o is not None and probs_o is not None:
            if sample and temperature > 0:
                idx = np.random.choice(len(moves_o), p=probs_o)
            else:
                idx = np.argmax(probs_o)
            move_o = moves_o[idx]
        
        return move_x, move_o, root