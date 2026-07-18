import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import height, width, board_size, device, num_parallel_games
from game import batched_lines_game
from model import alpha_lines_net


def generate_terminal_batch(num_games):
    """Plays target number of games to completion and returns raw terminal tensors."""
    env = batched_lines_game(num_games=num_games, device=device)
    
    while not env.finished.all():
        dist_p0 = torch.ones((num_games, height, width), device=device)
        dist_p1 = torch.ones((num_games, height, width), device=device)
        env.step(dist_p0, dist_p1)
        
    states_p0 = env.get_encoded_states(player=0)
    states_p1 = env.get_encoded_states(player=1)
    final_scores = env.scores.cpu()
    
    states_list = []
    values_list = []
    points_list = []
    
    for idx in range(num_games):
        s0 = final_scores[idx, 0].item()
        s1 = final_scores[idx, 1].item()
        
        # Player 0 Perspective
        states_list.append(states_p0[idx].clone())
        points_list.append(int(s0))
        if s0 > s1:    values_list.append(0)
        elif s0 == s1: values_list.append(1)
        else:          values_list.append(2)
            
        # Player 1 Perspective
        states_list.append(states_p1[idx].clone())
        points_list.append(int(s1))
        if s1 > s0:    values_list.append(0)
        elif s1 == s0: values_list.append(1)
        else:          values_list.append(2)
            
    states_t = torch.stack(states_list).to(device)
    values_t = torch.tensor(values_list, dtype=torch.long, device=device)
    points_t = torch.tensor(points_list, dtype=torch.long, device=device)
    points_t = torch.clamp(points_t, 0, board_size)
    
    # Apply Double Flip Augmentation
    states_flipped = torch.flip(states_t, dims=[-2, -1])
    
    train_states = torch.cat([states_t, states_flipped], dim=0)
    train_values = torch.cat([values_t, values_t], dim=0)
    train_points = torch.cat([points_t, points_t], dim=0)
    
    return train_states, train_values, train_points


def evaluate_and_print_single_game(model, iteration):
    """Plays one isolated game to completion and outputs clean tracking diagnostics."""
    env = batched_lines_game(num_games=1, device=device)
    
    while not env.finished.all():
        dist_p0 = torch.ones((1, height, width), device=device)
        dist_p1 = torch.ones((1, height, width), device=device)
        env.step(dist_p0, dist_p1)
        
    state_p0 = env.get_encoded_states(player=0)[0]
    s0 = int(env.scores[0, 0].item())
    s1 = int(env.scores[0, 1].item())
    
    if s0 > s1:    true_val_str = "Win (0)"
    elif s0 == s1: true_val_str = "Draw (1)"
    else:          true_val_str = "Loss (2)"
        
    model.eval()
    with torch.no_grad():
        _, value, value_error, points = model(state_p0.unsqueeze(0), apply_softmax=False)
        val_probs = F.softmax(value, dim=1)[0]
        pred_points = torch.argmax(points, dim=1)[0].item()
        pred_error = value_error[0].item()
        
    # Generate Visual ASCII Representation
    board_state = torch.argmax(state_p0[:5], dim=0)
    border = "+" + "-" * (width * 2) + "+"
    board_lines = [border]
    for r in range(height):
        line = "|"
        for c in range(width):
            v = board_state[r, c].item()
            if v == 0:    line += "  "  # non_playable_square
            elif v == 1:  line += "--"  # playable_square
            elif v == 2:  line += "##"  # removed_square
            elif v == 3:  line += "||"  # player_0_mark
            elif v == 4:  line += "oo"  # player_1_mark
        line += "|"
        board_lines.append(line)
    board_lines.append(border)
    board_ascii = "\n".join(board_lines)
    
    print("\n" + "=" * 75)
    print(f" LIVE DIAGNOSTIC MONITOR | ITERATION {iteration:03d}")
    print("=" * 75)
    print(board_ascii)
    print(f" Ground Truth Targets:     Scores: [P0: {s0}, P1: {s1}] | Outcome: {true_val_str}")
    print(f" Model Predicted Probs:    P(Win): {val_probs[0].item():.3f} | P(Draw): {val_probs[1].item():.3f} | P(Loss): {val_probs[2].item():.3f}")
    print(f" Model Scalar Prediction:  Points Head Argmax: {pred_points:<3} | Value Error Head: {pred_error:.4f}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    net = alpha_lines_net().to(device)
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    total_iterations = 50
    print(f"Initialized active training pipeline on: {device}")
    print(f"Running {total_iterations} iterations. (1 epoch per fresh batch, data discarded).")
    
    for iteration in range(1, total_iterations + 1):
        # 1. Generate new unique dataset and clear old buffers
        train_states, train_values, train_points = generate_terminal_batch(num_games=num_parallel_games)
        
        dataset = TensorDataset(train_states, train_values, train_points)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 2. Run precisely one epoch
        net.train()
        for batch_states, batch_values, batch_points in dataloader:
            opt.zero_grad()
            
            with torch.no_grad():
                _, val_probs, _, _ = net(batch_states, apply_softmax=True)
                v_pred = val_probs[:, 0] - val_probs[:, 2]
                v_true = torch.zeros_like(v_pred)
                v_true[batch_values == 0] = 1.0
                v_true[batch_values == 1] = 0.0
                v_true[batch_values == 2] = -1.0
                target_error = torch.abs(v_true - v_pred).unsqueeze(1)
            
            _, value, value_error, points = net(batch_states, apply_softmax=False)
            
            loss_value = F.cross_entropy(value, batch_values)
            loss_points = F.cross_entropy(points, batch_points)
            loss_error = F.mse_loss(value_error, target_error)
            
            total_loss = loss_value + loss_points + loss_error
            total_loss.backward()
            opt.step()
            
        # 3. Clean environment and run evaluation tracker
        evaluate_and_print_single_game(net, iteration)