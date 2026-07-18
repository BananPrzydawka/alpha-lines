import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import modal
from line_profiler import LineProfiler

# 1. Setup local directory and container image with local files and dependencies
local_dir = Path(__file__).parent
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "line-profiler")
    .add_local_dir(local_dir, remote_path="/root")
)

app = modal.App("alphalines-tests")

# 2. Local imports (resolved inside the container via the mounted directory)
from config import height, width, board_size, device, num_parallel_games
from game import batched_lines_game
from model import alpha_lines_net


def generate_future_mapped_batch(num_games):
    """
    Plays games to completion. Captures states AFTER the environment step
    to ensure the true terminal state (0 playable squares) is recorded.
    """
    env = batched_lines_game(num_games=num_games, device=device)
    game_histories = [[] for _ in range(num_games)]
    
    while not env.finished.all():
        active_mask = ~env.finished
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        
        dist_p0 = torch.ones((num_games, height, width), device=device)
        dist_p1 = torch.ones((num_games, height, width), device=device)
        env.step(dist_p0, dist_p1)
        
        states_p0 = env.get_encoded_states(player=0)
        states_p1 = env.get_encoded_states(player=1)
        
        for idx in active_indices:
            i = idx.item()
            game_histories[i].append((states_p0[i].clone(), states_p1[i].clone()))
        
    final_scores = env.scores.cpu()
    
    states_list = []
    values_list = []
    points_list = []
    
    for idx in range(num_games):
        s0 = int(final_scores[idx, 0].item())
        s1 = int(final_scores[idx, 1].item())
        
        val_p0 = 0 if s0 > s1 else (1 if s0 == s1 else 2)
        val_p1 = 0 if s1 > s0 else (1 if s1 == s0 else 2)
        
        last_two_steps = game_histories[idx][-2:]
        
        for state_p0, state_p1 in last_two_steps:
            states_list.append(state_p0)
            points_list.append(s0)
            values_list.append(val_p0)
                
            states_list.append(state_p1)
            points_list.append(s1)
            values_list.append(val_p1)
            
    states_t = torch.stack(states_list).to(device)
    values_t = torch.tensor(values_list, dtype=torch.long, device=device)
    points_t = torch.tensor(points_list, dtype=torch.long, device=device)
    points_t = torch.clamp(points_t, 0, board_size)
    
    states_flipped = torch.flip(states_t, dims=[-2, -1])
    
    train_states = torch.cat([states_t, states_flipped], dim=0)
    train_values = torch.cat([values_t, values_t], dim=0)
    train_points = torch.cat([points_t, points_t], dim=0)
    
    return train_states, train_values, train_points


def evaluate_and_print_future_sequence(model, iteration):
    """
    Evaluates the final 2 sequence steps of an isolated game. Displays current 
    vs. final terminal points, true value targets, and explicit value error targets.
    """
    env = batched_lines_game(num_games=1, device=device)
    history = []
    move_counts = []
    
    while not env.finished.all():
        dist_p0 = torch.ones((1, height, width), device=device)
        dist_p1 = torch.ones((1, height, width), device=device)
        env.step(dist_p0, dist_p1)
        
        state_p0 = env.get_encoded_states(player=0)[0].detach()
        history.append(state_p0)
        move_counts.append(env.move_counts[0].item())
        
    s0_final = int(env.scores[0, 0].item())
    s1_final = int(env.scores[0, 1].item())
    target_val = 0 if s0_final > s1_final else (1 if s0_final == s1_final else 2)
    
    v_true = 1.0 if target_val == 0 else (0.0 if target_val == 1 else -1.0)
    true_val_str = "Win (0)" if target_val == 0 else ("Draw (1)" if target_val == 1 else "Loss (2)")
        
    print("\n" + "#" * 85)
    print(f" FUTURE SEQUENCE MONITOR | ITERATION {iteration:03d}")
    print(f" Final Terminal Targets -> Scores: [P0: {s0_final}, P1: {s1_final}] | Outcome: {true_val_str}")
    print("#" * 85)

    model.eval()
    last_two_states = history[-2:]
    last_two_moves = move_counts[-2:]
    
    norm = width * height / 2
    
    for i, (state, step_num) in enumerate(zip(last_two_states, last_two_moves)):
        is_terminal = (i == 1)
        state_label = "TERMINAL STEP (T)" if is_terminal else "PENULTIMATE STEP (T-1)"
        
        s0_current = int(round(state[5, 0, 0].item() * norm))
        s1_current = int(round(state[6, 0, 0].item() * norm))
        
        with torch.no_grad():
            _, value, value_error, points = model(state.unsqueeze(0), apply_softmax=False)
            val_probs = F.softmax(value, dim=1)[0]
            pred_points = torch.argmax(points, dim=1)[0].item()
            pred_error = value_error[0].item()
            
        v_pred = val_probs[0].item() - val_probs[2].item()
        true_error_target = abs(v_true - v_pred)
            
        board_state = torch.argmax(state[:5], dim=0)
        border = "+" + "-" * (width * 2) + "+"
        board_lines = [border]
        for r in range(height):
            line = "|"
            for c in range(width):
                v = board_state[r, c].item()
                if v == 0:    line += "  "
                elif v == 1:  line += "--"
                elif v == 2:  line += "##"
                elif v == 3:  line += "||"
                elif v == 4:  line += "oo"
            line += "|"
            board_lines.append(line)
        board_lines.append(border)
        board_ascii = "\n".join(board_lines)
        
        m_win  = " <== [TRUE TARGET]" if target_val == 0 else ""
        m_draw = " <== [TRUE TARGET]" if target_val == 1 else ""
        m_loss = " <== [TRUE TARGET]" if target_val == 2 else ""
        m_pts  = " [MATCH]" if pred_points == s0_final else f" [MISMATCH | TRUE TARGET: {s0_final}]"
        
        lead_change_alert = ""
        if not is_terminal:
            current_leader = 0 if s0_current > s1_current else (1 if s0_current < s1_current else -1)
            final_leader = 0 if s0_final > s1_final else (1 if s0_final < s1_final else -1)
            if current_leader != final_leader and current_leader != -1 and final_leader != -1:
                lead_change_alert = " <== [LEAD CHANGE WILL OCCUR]"
        
        print(f"\n --- SEQUENCE STEP {i+1}/2 ({state_label} | move count: {step_num}) ---")
        print(board_ascii)
        print(f" State Scores At This Step:  [P0: {s0_current}, P1: {s1_current}]{lead_change_alert}")
        print(" Value Head Probability Estimation:")
        print(f"   P(Win):  {val_probs[0].item():.4f}{m_win}")
        print(f"   P(Draw): {val_probs[1].item():.4f}{m_draw}")
        print(f"   P(Loss): {val_probs[2].item():.4f}{m_loss}")
        print(" Regression Heads:")
        print(f"   Points Head Prediction (Argmax): {pred_points:<3}{m_pts}")
        print(f"   Value Error Head Prediction:    {pred_error:.4f}")
        print(f"   Value Error True Target:         {true_error_target:.4f}")
    print("=" * 85 + "\n")


def execute_training_loop():
    """
    Contains the core training execution logic to be targeted by LineProfiler.
    """
    net = alpha_lines_net().to(device)
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    total_iterations = 10
    print(f"Active training backend initialized on: {device}")
    
    for iteration in range(1, total_iterations + 1):
        print(f"\n>>> ITERATION {iteration:03d} | Simulating Games...")
        train_states, train_values, train_points = generate_future_mapped_batch(num_games=num_parallel_games)
        
        net.eval()
        with torch.no_grad():
            _, val_logits, _, _ = net(train_states, apply_softmax=False)
            preds = torch.argmax(val_logits, dim=1)
            
        true_wins   = (train_values == 0).sum().item()
        true_draws  = (train_values == 1).sum().item()
        true_losses = (train_values == 2).sum().item()
        
        pred_wins   = (preds == 0).sum().item()
        pred_draws  = (preds == 1).sum().item()
        pred_losses = (preds == 2).sum().item()
        
        misc_wins   = ((train_values == 0) & (preds != 0)).sum().item()
        misc_draws  = ((train_values == 1) & (preds != 1)).sum().item()
        misc_losses = ((train_values == 2) & (preds != 2)).sum().item()
        
        print("\n" + "-" * 62)
        print(f" TOTAL MISSCLASSIFIED: {misc_wins + misc_draws + misc_losses} | ITERATION {iteration:03d}")
        print("-" * 62)
        print(f" Outcome Class | True Count | Model Predicted | Misclassified")
        print(f" --------------|------------|-----------------|--------------")
        print(f" Win  (0)      | {true_wins:<10} | {pred_wins:<15} | {misc_wins}")
        print(f" Draw (1)      | {true_draws:<10} | {pred_draws:<15} | {misc_draws}")
        print(f" Loss (2)      | {true_losses:<10} | {pred_losses:<15} | {misc_losses}")
        print("-" * 62)
        
        dataset = TensorDataset(train_states, train_values, train_points)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        class_counts = torch.bincount(train_values, minlength=3).float()
        total_samples = train_values.numel()
        
        smoothed_weights = torch.sqrt(total_samples / class_counts.clamp(min=1.0))
        class_weights = smoothed_weights / smoothed_weights.mean()
        class_weights = class_weights.to(device)
        
        net.train()
        for b_idx, (batch_states, batch_values, batch_points) in enumerate(dataloader):
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
            loss_value = F.cross_entropy(value, batch_values, weight=class_weights)
            loss_points = F.cross_entropy(points, batch_points)
            loss_error = F.mse_loss(value_error, target_error)
            
            total_loss = loss_value + loss_points + loss_error
            total_loss.backward()
            opt.step()


# 3. Remote function designated to run inside the container with GPU resources
@app.function(image=image, gpu="RTX-PRO-6000", timeout=600)
def run_training():
    # --- RUN 1: Execution without LineProfiler ---
    print("\n" + "=" * 30 + " STARTING RUN: WITHOUT LINE PROFILER " + "=" * 30)
    start_no_prof = time.perf_counter()
    execute_training_loop()
    end_no_prof = time.perf_counter()
    duration_no_prof = end_no_prof - start_no_prof
    
    # --- RUN 2: Execution with LineProfiler ---
    print("\n" + "=" * 30 + " STARTING RUN: WITH LINE PROFILER " + "=" * 30)
    lp = LineProfiler()
    lp.add_function(generate_future_mapped_batch)
    lp_wrapper = lp(execute_training_loop)
    
    start_with_prof = time.perf_counter()
    lp_wrapper()
    end_with_prof = time.perf_counter()
    duration_with_prof = end_with_prof - start_with_prof
    
    # --- Calculate Overhead Metrics ---
    overhead_seconds = duration_with_prof - duration_no_prof
    overhead_percent = (overhead_seconds / duration_no_prof) * 100 if duration_no_prof > 0 else 0
    
    # --- Summary Report Outputs ---
    print("\n" + "=" * 42 + " OVERHEAD BENCHMARK SUMMARY " + "=" * 42)
    print(f" Execution Time WITHOUT Profiler : {duration_no_prof:.4f} seconds")
    print(f" Execution Time WITH Profiler    : {duration_with_prof:.4f} seconds")
    print(f" Absolute Profiler Overhead      : {overhead_seconds:.4f} seconds")
    print(f" Relative Overhead Percentage    : {overhead_percent:.2f}%")
    print("=" * 112 + "\n")
    
    # Print the detailed breakdown table collected during RUN 2
    print("\n" + "=" * 45 + " LINE PROFILER STATS " + "=" * 45)
    lp.print_stats(stream=sys.stdout)
    print("=" * 111 + "\n")


# 4. Local orchestration entrypoint
@app.local_entrypoint()
def main():
    print("Launching alpha-lines network training on Modal Cloud Cluster...")
    run_training.remote()