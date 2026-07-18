"""
Self-play game generation for Alpha-Lines with simultaneous moves
"""

import numpy as np
from game import LinesGame, flip_board_double, flip_policy_double, print_board
from mcts import MCTS
from config import (
    MCTS_SIMULATIONS, MAX_POINT_DIFFERENCE,
    BOARD_HEIGHT, BOARD_WIDTH
)


def play_game(model, device='cpu', temperature=1.0, sample=True, mcts_simulations=None, return_winner=False):
    """
    Play a complete game using self-play with MCTS.
    Both players make moves simultaneously.
    
    Args:
        return_winner: If True, returns (training_data, winner) tuple
    
    Returns:
        training_data: list of dicts with:
            - state: (6, 10, 16) numpy array (from X's perspective)
            - policy_x: (10, 16) numpy array (MCTS visit count distribution for X)
            - policy_o: (10, 16) numpy array (MCTS visit count distribution for O)
            - value: float (-1, 0, or 1) from X's perspective
            - point_diff: int (-80 to 80) point difference from X's perspective
        winner (if return_winner=True): 1 if X won, -1 if O won, 0 if draw
    """
    if mcts_simulations is None:
        mcts_simulations = MCTS_SIMULATIONS
    
    game = LinesGame()
    mcts = MCTS(model, device=device)
    
    # Store all positions from the game
    # Each entry stores state and policy targets for both players
    positions = []
    
    while not game.game_over:
        # Get MCTS actions for both players
        move_x, move_o, root = mcts.get_actions(
            game,
            num_simulations=mcts_simulations,
            temperature=temperature,
            sample=sample
        )
        
        if move_x is None and move_o is None:
            break
        
        # Get the policy distributions from MCTS
        moves_x, probs_x, moves_o, probs_o = root.get_action_distribution(temperature=temperature)
        
        # Create policy targets (10x16) for each player
        policy_x = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        policy_o = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        
        if moves_x is not None and probs_x is not None:
            for i, move in enumerate(moves_x):
                policy_x[move] = probs_x[i]
        
        if moves_o is not None and probs_o is not None:
            for i, move in enumerate(moves_o):
                policy_o[move] = probs_o[i]
        
        # Store position from X's perspective
        state_x = game.get_state_encoding(perspective=1)
        state_o = game.get_state_encoding(perspective=-1)
        
        positions.append({
            'state_x': state_x,
            'state_o': state_o,
            'policy_x': policy_x,
            'policy_o': policy_o,
        })
        
        # Execute both moves simultaneously
        game.execute_moves(move_x, move_o)
        # print(len(positions))
        # print_board(game)
    
    # Game is over, determine the outcome
    # Value is from X's perspective
    if game.winner == 1:  # X won
        value_for_x = 1.0
    elif game.winner == -1:  # O won
        value_for_x = -1.0
    else:  # Draw
        value_for_x = 0.0
    
    # Point difference from X's perspective
    point_diff = game.get_point_difference()
    # Clamp to [-80, 80]
    point_diff = max(-MAX_POINT_DIFFERENCE, min(MAX_POINT_DIFFERENCE, point_diff))
    
    # Create training data
    training_data = []
    
    for pos in positions:
        # Data from X's perspective
        training_data.append({
            'state': pos['state_x'],
            'policy': pos['policy_x'],
            'value': value_for_x,
            'point_diff': point_diff
        })
        
        # Data from O's perspective (swap marks, negate value and point_diff)
        state_o = pos['state_o'].copy()
        # Swap marks and scores (planes 2<->3, 4<->5)
        state_o[[2, 3]] = state_o[[3, 2]]
        state_o[[4, 5]] = state_o[[5, 4]]
        
        training_data.append({
            'state': state_o,
            'policy': pos['policy_o'],
            'value': -value_for_x,
            'point_diff': -point_diff
        })
    
    if return_winner:
        return training_data, game.winner
    return training_data


def augment_data(training_data):
    """
    Augment training data by double flip (horizontal + vertical).
    This preserves the diagonal geometry.
    This gives 2x the data from each position.
    """
    augmented = []
    
    for data in training_data:
        state = data['state']
        policy = data['policy']
        value = data['value']
        point_diff = data['point_diff']
        
        # Original
        augmented.append({
            'state': state,
            'policy': policy,
            'value': value,
            'point_diff': point_diff
        })
        
        # Double flip (horizontal + vertical)
        state_flipped = flip_board_double(state)
        policy_flipped = flip_policy_double(policy)
        
        augmented.append({
            'state': state_flipped,
            'policy': policy_flipped,
            'value': value,
            'point_diff': point_diff
        })
    
    return augmented


def generate_training_batch(model, device='cpu', target_positions=512, temperature=1.0):
    """
    Generate training data by playing games until we have enough positions.
    
    Args:
        model: The neural network model
        device: Device to run on
        target_positions: Number of unique positions to collect
        temperature: Temperature for move selection
    
    Returns:
        states: (N, 6, 10, 16) numpy array
        policies: (N, 10, 16) numpy array
        values: (N,) numpy array
        point_diff_classes: (N,) numpy array (int labels 0-160)
        leftover_data: List of positions that exceeded target (for next batch)
        game_stats: dict with 'wins', 'draws', 'losses' from X's perspective
    """
    all_data = []
    leftover_data = []
    
    # Track game outcomes
    x_wins = 0
    draws = 0
    x_losses = 0
    
    while len(all_data) < target_positions:
        # Play a game
        game_data, winner = play_game(model, device=device, temperature=temperature, return_winner=True)
        all_data.extend(game_data)
        
        # Track outcomes
        if winner == 1:
            x_wins += 1
        elif winner == 0:
            draws += 1
        else:
            x_losses += 1

    # Take exactly target_positions, save the rest for next batch
    leftover_count = len(all_data) - target_positions
    if leftover_count > 0:
        leftover_data = all_data[-leftover_count:]
        all_data = all_data[:target_positions]
    
    # Augment data (2x from double flip)
    augmented_data = augment_data(all_data)

    # Convert to arrays
    states = np.array([d['state'] for d in augmented_data])
    policies = np.array([d['policy'] for d in augmented_data])
    values = np.array([d['value'] for d in augmented_data])
    
    # Convert point difference to class index (0-160, where 80 = draw)
    point_diff_classes = np.array([
        int(d['point_diff'] + MAX_POINT_DIFFERENCE) for d in augmented_data
    ])
    
    game_stats = {
        'wins': x_wins,
        'draws': draws,
        'losses': x_losses,
        'total_games': x_wins + draws + x_losses
    }
    
    return states, policies, values, point_diff_classes, leftover_data, game_stats
