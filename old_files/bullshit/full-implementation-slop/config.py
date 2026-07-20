"""
Configuration and hyperparameters for Alpha-Lines
"""

# Game settings
BOARD_HEIGHT = 10 # default 10
BOARD_WIDTH = 16 # default 16
MAX_MOVES = 100  # Maximum number of moves before game ends
MAX_SCORE = 80  # Maximum possible score for a player
MAX_POINT_DIFFERENCE = 80  # Assumed max difference for one-hot encoding

# Neural network settings
NUM_RESIDUAL_BLOCKS = 20 # 20
NUM_FILTERS = 128 # 128
POLICY_HEAD_FILTERS = 2  # Filters for policy head conv
VALUE_HEAD_FILTERS = 1  # Filters for value head conv
POINT_HEAD_FILTERS = 1  # Filters for point predictor head conv

# Training settings
MINIBATCH_SIZE = 4096  # Total positions per minibatch (after augmentation)
UNIQUE_POSITIONS_PER_BATCH = 256  # Unique positions needed (* 2 augmentations)
LEARNING_RATE = 2e-2
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
NUM_EPOCHS_PER_BATCH = 1  # Epochs of training per self-play batch

# MCTS settings
MCTS_SIMULATIONS = 20 # 400 # Number of MCTS simulations per move during self-play
MCTS_C_PUCT = 1.0  # Exploration constant
MCTS_DIRICHLET_ALPHA = 0.03  # Dirichlet noise alpha
MCTS_DIRICHLET_WEIGHT = 0.25  # Weight of Dirichlet noise

# Evaluation settings
CHECKPOINT_INTERVAL = 2  # Save checkpoint every N batches
EVALUATION_GAMES = 32  # Number of games for elo evaluation
EVALUATION_MCTS_SIMULATIONS = 100  # MCTS simulations for evaluation
EVALUATION_TEMPERATURE = 0.5  # Temperature for sampling during evaluation (0 = deterministic, >0 = diverse)

# Elo settings
INITIAL_ELO = 0  # Starting ELO for the first model (anchored at 0)
ELO_K_FACTOR = 32  # K-factor for ELO updates
ELO_ANCHORED = True  # If True, only update the new model's Elo, old model stays fixed

# Data augmentation
# We use double flip (horizontal + vertical) which preserves diagonal geometry
# This gives us 2x data: original and flipped

# Paths
CHECKPOINT_DIR = "checkpoints"
ELO_LOG_FILE = "elo_history.json"

# Device
DEVICE = "cuda"  # Will be overridden if CUDA not available