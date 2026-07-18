import torch
import torch.nn.functional as F
import torch.optim as optim

from config import height, width, device, board_size
from game import batched_lines_game, print_game
from model import alpha_lines_net

# 1. Precompute explicit 1D action indices for width=16
# Top-Left (P0):     (0, 0)  -> 0 * 16 + 0  = 0
# Top-Right (P1):    (0, 14) -> 0 * 16 + 14 = 14
# Bottom-Left (P0):  (9, 1)  -> 9 * 16 + 1  = 145
# Bottom-Right (P1): (9, 15) -> 9 * 16 + 15 = 159
# Baseline (P0):     (0, 2)  -> 0 * 16 + 2  = 2
# Baseline (P1):     (0, 12) -> 0 * 16 + 12 = 12

idx_0 = torch.tensor([0, 145, 2], dtype=torch.long, device=device)
idx_1 = torch.tensor([12, 159, 12], dtype=torch.long, device=device)

# 2. Generate encodings through the game engine
env = batched_lines_game(num_games=3, device=device)
env.step_with_actions(idx_0, idx_1)

print_game(env, 0)
print_game(env, 1)
print_game(env, 2)

# Extract post-move state encodings from both perspectives
states_p0 = env.get_encoded_states(player=0)
states_p1 = env.get_encoded_states(player=1)

# 3. Formulate hardcoded target rules
# Game 0: Top corners -> Player 0 Win (Value 0 for P0, Value 2 for P1)
# Game 1: Bottom corners -> Player 1 Win (Value 2 for P0, Value 0 for P1)
# Game 2: Baseline -> Draw (Value 1 for both)
history_states = [
    states_p0[0], states_p0[1], states_p0[2],  # Player 0 perspectives
    states_p1[0], states_p1[1], states_p1[2]   # Player 1 perspectives
]

# Value mapping: Win = 0, Draw = 1, Loss = 2
target_values = torch.tensor([0, 2, 1, 2, 0, 1], dtype=torch.long, device=device)
target_errors = torch.zeros((6, 1), dtype=torch.float32, device=device)
target_points = torch.tensor([2, 0, 1, 0, 2, 1], dtype=torch.long, device=device)

all_states = torch.stack(history_states)

# 4. Apply Double Flip Augmentation (Simultaneous Vertical + Horizontal)
all_states_flipped = torch.flip(all_states, dims=[-2, -1])

train_states = torch.cat([all_states, all_states_flipped], dim=0)
train_values = torch.cat([target_values, target_values], dim=0)
train_errors = torch.cat([target_errors, target_errors], dim=0)
train_points = torch.cat([target_points, target_points], dim=0)

# train_states = all_states
# train_values = target_values
# train_errors = target_errors
# train_points = target_points

print(train_states[0][3])

# 5. Optimization Loop
model = alpha_lines_net().to(device)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

print("Starting game engine encoding validation test...")
print(f"Hardware backend: {device}")
print("-" * 60)

for epoch in range(1, 101):
    optimizer.zero_grad()
    
    policy, value, value_error, points = model(train_states, apply_softmax=False)
    
    loss_value = F.cross_entropy(value, train_values)
    loss_error = F.mse_loss(value_error, train_errors)
    loss_points = F.cross_entropy(points, train_points)
    
    total_loss = loss_value + loss_error + loss_points
    
    total_loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.6f} | Value Head: {loss_value.item():.6f} | Points Head: {loss_points.item():.6f}")

print("-" * 60)
print("Optimization complete. Evaluating convergence behavior on original perspectives:")

# 6. Evaluation Verification
model.eval()
with torch.no_grad():
    _, val_logits, err_pred, pt_logits = model(all_states, apply_softmax=False)
    val_probs = F.softmax(val_logits, dim=1)
    pt_preds = torch.argmax(pt_logits, dim=1)

perspective_names = [
    "Game 0 (Top Corners) - P0 Perspective",
    "Game 1 (Bottom Corners) - P0 Perspective",
    "Game 2 (Baseline) - P0 Perspective",
    "Game 0 (Top Corners) - P1 Perspective",
    "Game 1 (Bottom Corners) - P1 Perspective",
    "Game 2 (Baseline) - P1 Perspective"
]

for i in range(6):
    print(f"\n{perspective_names[i]}:")
    print(f"  Expected Value Class: {target_values[i].item()} | Predicted Probabilities: [Win: {val_probs[i, 0]:.4f}, Draw: {val_probs[i, 1]:.4f}, Loss: {val_probs[i, 2]:.4f}]")
    print(f"  Expected Point Class: {target_points[i].item()} | Predicted Point Class: {pt_preds[i].item()}")