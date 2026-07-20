"""
Main training script for Alpha-Lines

All hyperparameters are defined at the top of config.py
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Add the alpha-lines directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    # Game settings
    BOARD_HEIGHT, BOARD_WIDTH, MAX_MOVES, MAX_SCORE, MAX_POINT_DIFFERENCE,
    # Neural network settings
    NUM_RESIDUAL_BLOCKS, NUM_FILTERS,
    # Training settings
    MINIBATCH_SIZE, UNIQUE_POSITIONS_PER_BATCH, LEARNING_RATE,
    WEIGHT_DECAY, MOMENTUM, NUM_EPOCHS_PER_BATCH,
    # MCTS settings
    MCTS_SIMULATIONS, MCTS_C_PUCT, MCTS_DIRICHLET_ALPHA, MCTS_DIRICHLET_WEIGHT,
    # Training loop settings
    CHECKPOINT_INTERVAL, EVALUATION_GAMES, EVALUATION_MCTS_SIMULATIONS, EVALUATION_TEMPERATURE,
    # Elo settings
    INITIAL_ELO, ELO_K_FACTOR, ELO_ANCHORED,
    # Paths
    CHECKPOINT_DIR, ELO_LOG_FILE,
    # Device
    DEVICE
)

from model import AlphaLinesNet
from trainer import Trainer
from self_play import generate_training_batch
from elo import EloLadder


def print_hyperparameters():
    """Print all hyperparameters"""
    print("=" * 60)
    print("Alpha-Lines Training Configuration")
    print("=" * 60)
    print(f"Board size: {BOARD_HEIGHT}x{BOARD_WIDTH}")
    print(f"Max moves: {MAX_MOVES}")
    print(f"Max score: {MAX_SCORE}")
    print(f"Max point difference: {MAX_POINT_DIFFERENCE}")
    print(f"Residual blocks: {NUM_RESIDUAL_BLOCKS}")
    print(f"Filters: {NUM_FILTERS}")
    print(f"Minibatch size: {MINIBATCH_SIZE}")
    print(f"Unique positions per batch: {UNIQUE_POSITIONS_PER_BATCH}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Momentum: {MOMENTUM}")
    print(f"Epochs per batch: {NUM_EPOCHS_PER_BATCH}")
    print(f"MCTS simulations: {MCTS_SIMULATIONS}")
    print(f"MCTS C_PUCT: {MCTS_C_PUCT}")
    print(f"Dirichlet alpha: {MCTS_DIRICHLET_ALPHA}")
    print(f"Dirichlet weight: {MCTS_DIRICHLET_WEIGHT}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print(f"Evaluation games: {EVALUATION_GAMES}")
    print(f"Evaluation MCTS simulations: {EVALUATION_MCTS_SIMULATIONS}")
    print(f"Evaluation temperature: {EVALUATION_TEMPERATURE}")
    print(f"Initial ELO: {INITIAL_ELO}")
    print(f"ELO K-factor: {ELO_K_FACTOR}")
    print(f"ELO anchored: {ELO_ANCHORED}")
    print("=" * 60)


def main(num_batches=1000, resume_from=None):
    """
    Main training loop.
    
    Args:
        num_batches: Number of training batches to run
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Set device
    device = DEVICE if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    
    # Print hyperparameters
    print_hyperparameters()
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize model and trainer
    trainer = Trainer(device=device)
    
    # Initialize ELO ladder
    elo_ladder = EloLadder(log_file=os.path.join(CHECKPOINT_DIR, ELO_LOG_FILE))
    
    # Resume from checkpoint if specified
    start_batch = 0
    leftover_data = []
    old_model_path = None
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
        # Try to find batch number from filename
        try:
            start_batch = int(resume_from.split('_')[-1].replace('.pt', '')) + 1
        except:
            pass
    
    # Save initial model
    if len(elo_ladder.models) == 0:
        initial_path = os.path.join(CHECKPOINT_DIR, "model_0.pt")
        trainer.save_checkpoint(initial_path)
        elo_ladder.add_model(initial_path)
        print(f"Saved initial model: {initial_path}")
        old_model_path = initial_path
    
    # Training loop
    for batch_num in range(start_batch, num_batches):
        batch_start_time = time.time()
        
        # print(f"\n{'='*60}")
        # print(f"Batch {batch_num + 1}/{num_batches}")
        # print(f"{'='*60}")
        
        # Generate training data
        # print("Generating training data...")
        states, policies, values, point_diff_classes, leftover_data, game_stats = generate_training_batch(
            trainer.model,
            device=device,
            target_positions=UNIQUE_POSITIONS_PER_BATCH,
            temperature=1.0  # Temperature for exploration
        )
        
        print(f"X wins: {game_stats['wins']}, "
              f"Draws: {game_stats['draws']}, "
              f"X losses: {game_stats['losses']}")

        # print(f"Generated {len(states)} training positions (after augmentation)")
        
        # Train on the data
        # print("Training...")
        losses = trainer.train_batch(states, policies, values, point_diff_classes)
        
        batch_time = time.time() - batch_start_time
        
        print(f"Batch {batch_num + 1}/{num_batches}, Batch time: {batch_time:.2f}s")
        print(f"Losses - Policy: {losses['policy_loss']:.4f}, "
              f"Value: {losses['value_loss']:.4f}, "
              f"Point: {losses['point_loss']:.4f}, "
              f"Total: {losses['total_loss']:.4f}")
        # print(f"Batch time: {batch_time:.2f}s")
        
        # Save checkpoint and evaluate every CHECKPOINT_INTERVAL batches
        if (batch_num + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{batch_num + 1}.pt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            # Add to ELO ladder
            elo_ladder.add_model(checkpoint_path)
            
            # Evaluate against previous model
            if old_model_path is not None:
                print(f"Evaluating against previous model...")
                
                # Load old model for evaluation
                old_model = AlphaLinesNet().to(device)
                old_model.load_checkpoint(old_model_path, device=device)
                
                new_wins, draws, old_wins = elo_ladder.evaluate_models(
                    trainer.model, old_model, device=device
                )
                
                print(f"Evaluation results - New: {new_wins}, Draws: {draws}, Old: {old_wins}")
                
                # Update ELOs (with anchoring mode)
                # model_idx_1 = new model, model_idx_2 = old model
                # In ELO_ANCHORED mode, only the new model's Elo changes
                elo_ladder.record_match(
                    len(elo_ladder.models) - 1,  # New model index
                    len(elo_ladder.models) - 2,  # Old model index (anchored if ELO_ANCHORED)
                    new_wins, draws, old_wins
                )
                
                if ELO_ANCHORED:
                    print(f"ELO anchoring enabled - old model Elo remains unchanged")
                
                # Log detailed evaluation
                elo_ladder.log_evaluation(
                    batch_num + 1,
                    new_wins, draws, old_wins,
                    elo_ladder.models[-1][1],
                    elo_ladder.models[-2][1]
                )
                
                elo_ladder.print_ladder()
                
                old_model_path = checkpoint_path
            else:
                old_model_path = checkpoint_path
        
        # Print progress
        # if (batch_num + 1) % 10 == 0:
        #     print(f"\nProgress: {batch_num + 1}/{num_batches} batches completed")
    
    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Print final ELO ladder
    elo_ladder.print_ladder()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Alpha-Lines model")
    parser.add_argument("--batches", type=int, default=1000, help="Number of training batches")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    main(num_batches=args.batches, resume_from=args.resume)