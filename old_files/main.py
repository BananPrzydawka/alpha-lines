import torch
import time
torch.set_printoptions(linewidth=float(1000))

from config import height, width, board_size, filters, bottleneck, resblock_number, policy_filters, value_fc, error_fc, point_fc, num_parallel_games, mcts_num_simulations, device
from game import lines_game, batched_lines_game, print_game_from_encoding
from model import alpha_lines_net
from monte_carlo import BatchedMCTS
# from mock_model import alpha_lines_net

# Instantiate network and batched engines
model = alpha_lines_net().to(device)
model.eval()

games = batched_lines_game(num_games=num_parallel_games, device=device)
mcts_engine = BatchedMCTS(model=model, device=device)

while not games.finished.all():

    # 1. Complete parallel MCTS search routines for all concurrent lines
    p0_targets, p1_targets = mcts_engine.search(games)
    
    # print(model((games.get_encoded_states(0))[0]))
    # print("p0 mcts dist:")
    # print(p0_targets[0])
    # print("p1 mcts dist")
    # print(p1_targets[0])

    # 2. Extract decisions via search distributions (e.g., sample or argmax)
    # Flattens distributions for multinomial sampling
    flat_p0 = p0_targets.view(num_parallel_games, -1)
    flat_p1 = p1_targets.view(num_parallel_games, -1)
    
    # Handle fully completed game lanes to bypass sampling errors
    flat_p0[games.finished] = 1.0 / board_size
    flat_p1[games.finished] = 1.0 / board_size
    
    sampled_a0 = torch.multinomial(flat_p0, 1).squeeze(1)
    sampled_a1 = torch.multinomial(flat_p1, 1).squeeze(1)
    
    # 3. Advance the master environment states deterministically
    games.step_with_actions(sampled_a0, sampled_a1)
    
    # 4. Advance trees to reuse information, discarding unchosen options
    if not games.finished.all():
        mcts_engine.advance_roots(games, sampled_a0, sampled_a1)
    
    print_game_from_encoding(games, index=0)

    # time.sleep(1.5)