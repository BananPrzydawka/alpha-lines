"""
Elo evaluation system for Alpha-Lines
"""

import json
import os
import numpy as np
from game import LinesGame, print_board
from mcts import MCTS
from config import (
    EVALUATION_GAMES, EVALUATION_MCTS_SIMULATIONS, EVALUATION_TEMPERATURE,
    INITIAL_ELO, ELO_K_FACTOR, ELO_ANCHORED
)


class EloLadder:
    """
    Tracks ELO ratings for different model checkpoints.
    """
    
    def __init__(self, log_file=None):
        self.log_file = log_file or "elo_history.json"
        self.models = []  # List of [checkpoint_path, elo_rating]
        self.history = []  # List of evaluation results
        
        # Load existing history if available
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                # Keep as list of lists (mutable)
                self.models = [[m['path'], m['elo']] for m in data.get('models', [])]
                self.history = data.get('history', [])
    
    def add_model(self, checkpoint_path):
        """Add a new model to the ladder with initial ELO"""
        if len(self.models) == 0:
            elo = INITIAL_ELO
        else:
            # Start with the same ELO as the latest model
            elo = self.models[-1][1]
        
        # Store as list for mutability in record_game
        self.models.append([checkpoint_path, elo])
        self._save()
        return elo
    
    def record_match(self, model_idx_1, model_idx_2, wins, draws, losses):
        """
        Record a match between two models and update ELOs.
        
        Args:
            model_idx_1: Index of first model in self.models (typically the new model)
            model_idx_2: Index of second model (typically the old/reference model)
            wins: Number of games model 1 won
            draws: Number of drawn games
            losses: Number of games model 1 lost (model 2 won)
        """
        elo_1 = self.models[model_idx_1][1]
        elo_2 = self.models[model_idx_2][1]
        
        total_games = wins + draws + losses
        
        # Expected score for model 1
        expected_1 = 1.0 / (1.0 + 10 ** ((elo_2 - elo_1) / 400))
        
        # Actual score for model 1 (wins + 0.5 * draws) / total_games
        actual_1 = (wins + 0.5 * draws) / total_games
        
        if ELO_ANCHORED:
            # Only update model_idx_1 (the new model), keep model_idx_2 (old model) anchored
            new_elo_1 = elo_1 + ELO_K_FACTOR * (actual_1 - expected_1)
            self.models[model_idx_1][1] = new_elo_1
            # model_idx_2 stays unchanged (anchored)
        else:
            # Update both ELOs (traditional Elo)
            expected_2 = 1.0 - expected_1
            actual_2 = 1.0 - actual_1
            new_elo_1 = elo_1 + ELO_K_FACTOR * (actual_1 - expected_1)
            new_elo_2 = elo_2 + ELO_K_FACTOR * (actual_2 - expected_2)
            self.models[model_idx_1][1] = new_elo_1
            self.models[model_idx_2][1] = new_elo_2
        
        self._save()
    
    def record_game(self, model_idx_1, model_idx_2, result):
        """
        Deprecated: Use record_match instead for proper aggregate scoring.
        Record a single game result and update ELOs.
        
        Args:
            model_idx_1: Index of first model in self.models (typically the new model)
            model_idx_2: Index of second model (typically the old/reference model)
            result: 1 if model 1 won, 0 if draw, -1 if model 2 won
        """
        elo_1 = self.models[model_idx_1][1]
        elo_2 = self.models[model_idx_2][1]
        
        # Expected scores
        expected_1 = 1.0 / (1.0 + 10 ** ((elo_2 - elo_1) / 400))
        expected_2 = 1.0 - expected_1
        
        # Actual scores
        if result == 1:
            actual_1 = 1.0
            actual_2 = 0.0
        elif result == -1:
            actual_1 = 0.0
            actual_2 = 1.0
        else:
            actual_1 = 0.5
            actual_2 = 0.5
        
        if ELO_ANCHORED:
            # Only update model_idx_1 (the new model), keep model_idx_2 (old model) anchored
            new_elo_1 = elo_1 + ELO_K_FACTOR * (actual_1 - expected_1)
            self.models[model_idx_1][1] = new_elo_1
            # model_idx_2 stays unchanged (anchored)
        else:
            # Update both ELOs (traditional Elo)
            new_elo_1 = elo_1 + ELO_K_FACTOR * (actual_1 - expected_1)
            new_elo_2 = elo_2 + ELO_K_FACTOR * (actual_2 - expected_2)
            self.models[model_idx_1][1] = new_elo_1
            self.models[model_idx_2][1] = new_elo_2
        
        self._save()
    
    def evaluate_models(self, model_new, model_old, device='cpu'):
        """
        Play evaluation games between two models.
        Each game has both models playing simultaneously.
        
        Args:
            model_new: New model to evaluate
            model_old: Old model to compare against
            device: Device to run on
        
        Returns:
            (new_wins, draws, old_wins)
        """
        new_wins = 0
        draws = 0
        old_wins = 0
        
        for game_num in range(EVALUATION_GAMES):
            # Alternate who plays as X (first player)
            if game_num % 2 == 0:
                new_is_x = True
            else:
                new_is_x = False
            
            result = self._play_evaluation_game(
                model_new if new_is_x else model_old,
                model_old if new_is_x else model_new,
                device=device
            )
            
            if result == 1:  # X won
                if new_is_x:
                    new_wins += 1
                else:
                    old_wins += 1
            elif result == -1:  # O won
                if new_is_x:
                    old_wins += 1
                else:
                    new_wins += 1
            else:
                draws += 1
        
        return new_wins, draws, old_wins
    
    def _play_evaluation_game(self, model_x, model_o, device='cpu'):
        """
        Play a single evaluation game with simultaneous moves.
        Returns: 1 if X wins, -1 if O wins, 0 if draw
        """
        game = LinesGame()
        
        while not game.game_over:
            # Get actions for both players using their respective models
            # For evaluation, we use sampling with EVALUATION_TEMPERATURE for diversity
            move_x, move_o, _ = self._get_simultaneous_actions(
                game, model_x, model_o, device=device
            )
            
            if move_x is None and move_o is None:
                break
            
            game.execute_moves(move_x, move_o)
            print_board(game)
        
        return game.winner
    
    def _get_simultaneous_actions(self, game, model_x, model_o, device='cpu'):
        """
        Get actions for both players using their respective models.
        Uses sampling with EVALUATION_TEMPERATURE for diverse games.
        """
        # Run MCTS for X using model_x
        mcts_x = MCTS(model_x, device=device)
        root_x = mcts_x.search(game, player=1, num_simulations=EVALUATION_MCTS_SIMULATIONS,
                               add_dirichlet_noise=False)
        
        # Run MCTS for O using model_o
        mcts_o = MCTS(model_o, device=device)
        root_o = mcts_o.search(game, player=-1, num_simulations=EVALUATION_MCTS_SIMULATIONS,
                               add_dirichlet_noise=False)
        
        move_x = None
        move_o = None
        
        # X's action from root_x with sampling
        moves_x, probs_x, _, _ = root_x.get_action_distribution(temperature=EVALUATION_TEMPERATURE)
        if moves_x is not None and probs_x is not None:
            if EVALUATION_TEMPERATURE > 0:
                idx = np.random.choice(len(moves_x), p=probs_x)
            else:
                idx = np.argmax(probs_x)
            move_x = moves_x[idx]
        
        # O's action from root_o with sampling
        _, _, moves_o, probs_o = root_o.get_action_distribution(temperature=EVALUATION_TEMPERATURE)
        if moves_o is not None and probs_o is not None:
            if EVALUATION_TEMPERATURE > 0:
                idx = np.random.choice(len(moves_o), p=probs_o)
            else:
                idx = np.argmax(probs_o)
            move_o = moves_o[idx]
        
        return move_x, move_o, root_x
    
    def get_latest_elo(self):
        """Get the ELO of the latest model"""
        if len(self.models) == 0:
            return INITIAL_ELO
        return self.models[-1][1]
    
    def get_elo_history(self):
        """Get the ELO history of all models"""
        return [(path, elo) for path, elo in self.models]
    
    def _save(self):
        """Save ELO data to file"""
        data = {
            'models': [{'path': m[0], 'elo': m[1]} for m in self.models],
            'history': self.history
        }
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_evaluation(self, batch_num, new_wins, draws, old_wins, new_elo, old_elo):
        """Log an evaluation result"""
        entry = {
            'batch': batch_num,
            'new_wins': new_wins,
            'draws': draws,
            'old_wins': old_wins,
            'new_elo': new_elo,
            'old_elo': old_elo
        }
        self.history.append(entry)
        self._save()
    
    def print_ladder(self):
        """Print the current ELO ladder"""
        print("\n=== ELO LADDER ===")
        for i, (path, elo) in enumerate(self.models):
            print(f"Model {i}: {path} - ELO: {elo:.1f}")
        print("==================\n")