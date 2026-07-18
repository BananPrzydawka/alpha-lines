import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
        
        # Capture encodings immediately AFTER the step executes
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
    
    # Value scalar target: Win=1.0, Draw=0.0, Loss=-1.0
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
        
        # Extract current state score configuration directly from plane 5 and 6
        s0_current = int(round(state[5, 0, 0].item() * norm))
        s1_current = int(round(state[6, 0, 0].item() * norm))
        
        with torch.no_grad():
            _, value, value_error, points = model(state.unsqueeze(0), apply_softmax=False)
            val_probs = F.softmax(value, dim=1)[0]
            pred_points = torch.argmax(points, dim=1)[0].item()
            pred_error = value_error[0].item()
            
        # Compute exact value error target based on current model outputs
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
        
        # Check if the leading player changes during the last step transition
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


if __name__ == "__main__":
    net = alpha_lines_net().to(device)
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    total_iterations = 5000
    print(f"Active training backend initialized on: {device}")
    
    for iteration in range(1, total_iterations + 1):
        print(f"\n>>> ITERATION {iteration:03d} | Simulating Games...")
        train_states, train_values, train_points = generate_future_mapped_batch(num_games=num_parallel_games)
        
        dataset = TensorDataset(train_states, train_values, train_points)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
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
            
            loss_value = F.cross_entropy(value, batch_values)
            loss_points = F.cross_entropy(points, batch_points)
            loss_error = F.mse_loss(value_error, target_error)
            
            total_loss = loss_value + loss_points + loss_error
            total_loss.backward()
            opt.step()
            
            # print(f"    Batch {b_idx+1:02d}/{len(dataloader):02d} -> Value Head Loss: {loss_value.item():.4f} | Points Head Loss: {loss_points.item():.4f} | Error Head Loss: {loss_error.item():.4f}")
            
        evaluate_and_print_future_sequence(net, iteration)