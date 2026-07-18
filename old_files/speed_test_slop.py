import torch
import time
from game import batched_lines_game
# from model import alpha_lines_net
# from game_gemini import lines_game


# Initialize model and move to device
# model = alpha_lines_net().to(device)
# model.eval()

# def test_random_base_serial(num_games):
#     start_time = time.time()
#     policy_dummy = torch.ones(10, 16)
    
#     for _ in range(num_games):
#         game = lines_game()
#         while not game.finished:
#             game.make_move_from_distributions(policy_dummy, policy_dummy)
            
#     return time.time() - start_time

# def test_random_base_batched(num_games):
#     games = [lines_game() for _ in range(num_games)]
#     active_indices = list(range(num_games))
#     start_time = time.time()
    
#     while active_indices:
#         batch_size = len(active_indices)
#         # Simulate batch generation
#         policies_0 = torch.ones(batch_size, 10, 16)
#         policies_1 = torch.ones(batch_size, 10, 16)
        
#         next_active_indices = []
#         for batch_idx, game_idx in enumerate(active_indices):
#             game = games[game_idx]
#             game.make_move_from_distributions(policies_0[batch_idx], policies_1[batch_idx])
#             if not game.finished:
#                 next_active_indices.append(game_idx)
#         active_indices = next_active_indices
        
#     return time.time() - start_time

# def test_random_batched_serial(num_games):
#     start_time = time.time()

#     for _ in range(num_games):
#         game = batched_lines_game()
#         policy_dummy = torch.ones(10, 16)
#         while not game.finished:
#             game.step(policy_dummy, policy_dummy)

#     return time.time() - start_time

def test_random_batched_batched(num_games):
    start_time = time.time()
    bg = batched_lines_game(num_games)

    while not bg.finished.all():
        dist = torch.ones(num_games, 10, 16)
        bg.distribution_step(dist, dist)

    return time.time() - start_time

# def test_model_serial(num_games):
#     """Plays games sequentially using model inference (GPU)."""
#     start_time = time.time()
    
#     with torch.no_grad():
#         for _ in range(num_games):
#             game = lines_game()
#             while not game.finished:
#                 # Encode and add batch dimension
#                 state_0 = game.get_encoded_state(player=0).unsqueeze(0).to(device)
#                 state_1 = game.get_encoded_state(player=0).unsqueeze(0).to(device)
                
#                 # Inference
#                 p0, _, _, _ = model.forward(state_0, True)
#                 p1, _, _, _ = model.forward(state_1, True)
                
#                 # Move back to CPU and remove batch dimension for the game engine
#                 game.make_move_from_distributions(p0[0].cpu(), p1[0].cpu())
                
#     if device.type == "cuda":
#         torch.cuda.synchronize()
#     return time.time() - start_time

# def test_model_batched(num_games):
#     """Plays games in parallel using model inference (GPU)."""
#     games = [lines_game() for _ in range(num_games)]
#     active_indices = list(range(num_games))
#     start_time = time.time()
    
#     with torch.no_grad():
#         while active_indices:
#             # Gather states from active games
#             states_0 = torch.stack([games[i].get_encoded_state(player=0) for i in active_indices]).to(device)
#             states_1 = torch.stack([games[i].get_encoded_state(player=0) for i in active_indices]).to(device)
            
#             # Batch Inference
#             p0, _, _, _ = model.forward(states_0, True)
#             p1, _, _, _ = model.forward(states_1, True)
            
#             p0, p1 = p0.cpu(), p1.cpu()
            
#             next_active_indices = []
#             for batch_idx, game_idx in enumerate(active_indices):
#                 game = games[game_idx]
#                 game.make_move_from_distributions(p0[batch_idx], p1[batch_idx])
#                 if not game.finished:
#                     next_active_indices.append(game_idx)
#             active_indices = next_active_indices
            
#     if device.type == "cuda":
#         torch.cuda.synchronize()
#     return time.time() - start_time

def main():
    # warmup
    test_random_batched_batched(100)

    for number in [1, 4, 16, 64, 256, 1024, 4096]:

        total_time = test_random_batched_batched(number)

        # print(f"{test_random_base_serial(NUM_GAMES):.4f}s")
        # print(f"{test_random_base_batched(NUM_GAMES):.4f}s")
        # print(f"{test_random_batched_serial(NUM_GAMES):.4f}s")
        print(f"testing games: {number} | total time: {total_time:.3f}s | per game: {total_time/number*1000:.3f}ms")


        # t_model_serial = test_model_serial(NUM_GAMES)
        # print(f"Model Serial:   {t_model_serial:.4f}s")

        # t_model_batch = test_model_batched(NUM_GAMES)
        # print(f"Model Batched:  {t_model_batch:.4f}s")