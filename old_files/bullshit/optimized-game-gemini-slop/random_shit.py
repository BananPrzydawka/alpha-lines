import torch
import time
from game import lines_game
# from model import alpha_lines_net

# --- Configuration ---
NUM_GAMES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move to device
# model = alpha_lines_net().to(device)
# model.eval()

def test_random_serial(num_games):
    """Plays games sequentially using constant tensors (CPU)."""
    start_time = time.time()
    policy_dummy = torch.ones(10, 16)
    
    for _ in range(num_games):
        game = lines_game()
        while not game.finished:
            game.make_move_from_distributions(policy_dummy, policy_dummy)
            
    return time.time() - start_time

def test_random_batched(num_games):
    """Plays games in parallel using constant tensors (CPU)."""
    games = [lines_game() for _ in range(num_games)]
    active_indices = list(range(num_games))
    start_time = time.time()
    
    while active_indices:
        batch_size = len(active_indices)
        # Simulate batch generation
        policies_0 = torch.ones(batch_size, 10, 16)
        policies_1 = torch.ones(batch_size, 10, 16)
        
        next_active_indices = []
        for batch_idx, game_idx in enumerate(active_indices):
            game = games[game_idx]
            game.make_move_from_distributions(policies_0[batch_idx], policies_1[batch_idx])
            if not game.finished:
                next_active_indices.append(game_idx)
        active_indices = next_active_indices
        
    return time.time() - start_time

# def test_model_batched(num_games):
    # """Plays games in parallel using model inference (GPU)."""
    # games = [lines_game() for _ in range(num_games)]
    # active_indices = list(range(num_games))
    # start_time = time.time()
    
    # with torch.no_grad():
    #     while active_indices:
    #         # Gather states from active games
    #         states_0 = torch.stack([games[i].get_encoded_state(player=0) for i in active_indices]).to(device)
    #         states_1 = torch.stack([games[i].get_encoded_state(player=0) for i in active_indices]).to(device)
            
    #         # Batch Inference
    #         p0, _, _, _ = model.forward(states_0, True)
    #         p1, _, _, _ = model.forward(states_1, True)
            
    #         p0, p1 = p0.cpu(), p1.cpu()
            
    #         next_active_indices = []
    #         for batch_idx, game_idx in enumerate(active_indices):
    #             game = games[game_idx]
    #             game.make_move_from_distributions(p0[batch_idx], p1[batch_idx])
    #             if not game.finished:
    #                 next_active_indices.append(game_idx)
    #         active_indices = next_active_indices
            
    # if device.type == "cuda":
    #     torch.cuda.synchronize()
    # return time.time() - start_time

# --- Execution and Results ---
print(f"Testing {NUM_GAMES} games | Device: {device}\n" + "-"*40)

t_rand_serial = test_random_serial(NUM_GAMES)
t_rand_batch = test_random_batched(NUM_GAMES)
# t_model_batch = test_model_batched(NUM_GAMES)

print(f"Random Serial:  {t_rand_serial:.4f}s")
print(f"Random Batched: {t_rand_batch:.4f}s")
# print(f"Model Batched:  {t_model_batch:.4f}s")