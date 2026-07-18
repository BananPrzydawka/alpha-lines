import torch
import numpy as np
import random
from config import height, width

# Import both versions
import game as opt
import game_slop as slop

def to_tensor(data):
    """Safely converts list of lists or tensors to a torch.int8 tensor."""
    if isinstance(data, torch.Tensor):
        return data.detach().clone().to(torch.int8)
    return torch.tensor(data, dtype=torch.int8)

def compare_states(g_opt, g_slop):
    """Compares the current state of both game instances."""
    # Normalize boards to tensors for comparison
    board_opt = to_tensor(g_opt.board)
    board_slop = to_tensor(g_slop.board)
    
    assert torch.equal(board_opt, board_slop), "Boards do not match"
    assert list(g_opt.scores) == list(g_slop.scores), f"Scores do not match: {g_opt.scores} vs {g_slop.scores}"
    assert g_opt.move_count == g_slop.move_count, "Move counts do not match"
    assert g_opt.finished == g_slop.finished, "Finished status does not match"

    # Compare legal move masks (normalized to tensors)
    mask_opt_0 = to_tensor(g_opt.get_legal_move_mask(0))
    mask_slop_0 = to_tensor(g_slop.get_legal_move_mask(0))
    assert torch.equal(mask_opt_0, mask_slop_0), "Legal move masks for Player 0 do not match"

    # Compare encoded states
    enc_opt_0 = g_opt.get_encoded_state(0)
    enc_slop_0 = g_slop.get_encoded_state(0)
    assert torch.allclose(enc_opt_0, enc_slop_0), "Encoded states for Player 0 do not match"

def run_test():
    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    g_opt = opt.lines_game()
    g_slop = slop.lines_game()

    print(f"Testing {height}x{width} board...")

    # 1. Initial State Check
    compare_states(g_opt, g_slop)
    print("Initial state check passed.")

    # 2. First Move Check (Restricted half-board)
    mask_0 = to_tensor(g_opt.get_legal_move_mask(0))
    mask_1 = to_tensor(g_opt.get_legal_move_mask(1))
    
    idx_0 = (mask_0 == 1).nonzero(as_tuple=True)
    idx_1 = (mask_1 == 1).nonzero(as_tuple=True)
    
    # Select first available legal move
    m0 = (idx_0[0][0].item(), idx_0[1][0].item())
    m1 = (idx_1[0][0].item(), idx_1[1][0].item())

    g_opt.make_move(m0, m1)
    g_slop.make_move(m0, m1)
    compare_states(g_opt, g_slop)
    print("First move check passed.")

    # 3. Collision Check
    g_opt = opt.lines_game()
    g_slop = slop.lines_game()
    
    # Force a collision on a valid playable square
    collision_move = (0, 0) if (0 + 0) % 2 == 0 else (0, 1)
         
    g_opt.make_move(collision_move, collision_move)
    g_slop.make_move(collision_move, collision_move)
    compare_states(g_opt, g_slop)
    print("Collision logic check passed.")

    # 4. Random Playthrough
    print("Starting random playthrough...")
    g_opt = opt.lines_game()
    g_slop = slop.lines_game()

    while not g_opt.finished:
        mask_0 = to_tensor(g_opt.get_legal_move_mask(0))
        mask_1 = to_tensor(g_opt.get_legal_move_mask(1))
        
        valid_0 = mask_0.nonzero()
        valid_1 = mask_1.nonzero()
        
        if len(valid_0) == 0 or len(valid_1) == 0:
            break
            
        m0 = tuple(valid_0[random.randrange(len(valid_0))].tolist())
        m1 = tuple(valid_1[random.randrange(len(valid_1))].tolist())
        
        g_opt.make_move(m0, m1)
        g_slop.make_move(m0, m1)
        compare_states(g_opt, g_slop)

    print(f"Random playthrough passed. Final Score: {g_opt.scores}")

if __name__ == "__main__":
    try:
        run_test()
        print("\nSUCCESS: Both implementations are functionally identical.")
    except AssertionError as e:
        print(f"\nFAILURE: {e}")