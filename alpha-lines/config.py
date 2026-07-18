import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# uv run --with tensorboard --with torch-tb-profiler tensorboard --logdir=./alpha-lines-folders/logs/tb_logs/

gpu = "B300"
timeout = 600
host_logs = "./logs"
log_interval = 10
game_kernels_parralel = False

height = 10                 # 10
width = 16                  # 16
board_size = height * width

iterations = 1
num_parallel_games = 512
batch_size = 2048
learning_rate = 7e-4
weight_decay = 1e-4

mcts_num_simulations = 10
mcts_c_puct = 1.5
mcts_alpha = 0.25
mcts_epsilon = 0.25
mcts_epx3_gamma = 0.1

filters = 128               # 256 in alpha zero
bottleneck = 32             # 32 in lc0
resblock_number = 30        # 40 in alpha zero

policy_filters = 60         # 80 in lc0
value_fc = 256
error_fc = 128
point_fc = 256