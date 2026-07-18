import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import height, width, board_size, device, num_parallel_games, batch_size, learning_rate, weight_decay, iterations
from game import batched_lines_game
from model import alpha_lines_net


def generate_future_mapped_batch(num_games):
    """
    Plays games to completion. Captures states AFTER the environment step
    to ensure the true terminal state (0 playable squares) is recorded.
    """
    env = batched_lines_game(num_games=num_games, device=device)
    init_p0 = env.get_encoded_states(player=0)
    state_shape = init_p0.shape[1:]
    
    penultimate_p0 = torch.zeros((num_games, *state_shape), device=device)
    penultimate_p1 = torch.zeros((num_games, *state_shape), device=device)
    terminal_p0 = torch.zeros((num_games, *state_shape), device=device)
    terminal_p1 = torch.zeros((num_games, *state_shape), device=device)
    game_step_counts = torch.zeros(num_games, dtype=torch.long, device=device)
    
    while not env.finished.all():
        # env.print_game(0)

        active_mask = ~env.finished
        
        dist_p0 = torch.ones((num_games, height, width), device=device)
        dist_p1 = torch.ones((num_games, height, width), device=device)
        env.distribution_step(dist_p0, dist_p1)
        
        states_p0 = env.get_encoded_states(player=0)
        states_p1 = env.get_encoded_states(player=1)
        
        penultimate_p0[active_mask] = terminal_p0[active_mask]
        penultimate_p1[active_mask] = terminal_p1[active_mask]
        terminal_p0[active_mask] = states_p0[active_mask]
        terminal_p1[active_mask] = states_p1[active_mask]
        game_step_counts[active_mask] += 1
        
    s0_all = env.scores[:, 0].long()
    s1_all = env.scores[:, 1].long()
    val_p0_all = torch.where(s0_all > s1_all, 0, torch.where(s0_all == s1_all, 1, 2))
    val_p1_all = torch.where(s1_all > s0_all, 0, torch.where(s1_all == s0_all, 1, 2))
    
    two_steps_mask = game_step_counts >= 2
    p_states = torch.cat([penultimate_p0[two_steps_mask], penultimate_p1[two_steps_mask]], dim=0)
    p_points = torch.cat([s0_all[two_steps_mask], s1_all[two_steps_mask]], dim=0)
    p_values = torch.cat([val_p0_all[two_steps_mask], val_p1_all[two_steps_mask]], dim=0)
    
    t_states = torch.cat([terminal_p0, terminal_p1], dim=0)
    t_points = torch.cat([s0_all, s1_all], dim=0)
    t_values = torch.cat([val_p0_all, val_p1_all], dim=0)
    
    states_t = torch.cat([p_states, t_states], dim=0)
    values_t = torch.cat([p_values, t_values], dim=0)
    points_t = torch.cat([p_points, t_points], dim=0)
    points_t = torch.clamp(points_t, 0, board_size)
    
    states_flip_both = torch.flip(states_t, dims=[-2, -1])
    states_flip_horiz = torch.flip(states_t, dims=[-1])
    states_flip_vert = torch.flip(states_t, dims=[-2])
    
    train_states = torch.cat([states_t, states_flip_both, states_flip_horiz, states_flip_vert], dim=0)
    train_values = torch.cat([values_t, values_t, values_t, values_t], dim=0)
    train_points = torch.cat([points_t, points_t, points_t, points_t], dim=0)
    
    return train_states, train_values, train_points


def execute_training_loop(prof=None):
    """
    Contains the core training execution logic, optimized via global GPU calculations.
    Accepts an optional torch.profiler object to record structural steps.
    """
    torch.set_float32_matmul_precision('high')
    net = alpha_lines_net().to(device)
    opt = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    total_iterations = iterations
    print(f"Active training backend initialized on: {device}")
    
    for iteration in range(1, total_iterations + 1):
        t0 = time.time()
        print(f"\n>>> ITERATION {iteration:03d} | Simulating Games...")
        train_states, train_values, train_points = generate_future_mapped_batch(num_games=num_parallel_games)
        
        class_counts = torch.bincount(train_values, minlength=3).float()
        total_samples = train_values.numel()
        
        smoothed_weights = torch.sqrt(total_samples / class_counts.clamp(min=1.0))
        class_weights = (smoothed_weights / smoothed_weights.mean()).to(device)
        indices = torch.randperm(total_samples, device=device)
        
        true_counts = torch.bincount(train_values, minlength=3)
        running_true_wins = true_counts[0].item()
        running_true_draws = true_counts[1].item()
        running_true_losses = true_counts[2].item()
        
        all_preds = torch.zeros(total_samples, dtype=torch.long, device=device)
        
        net.train()
        for start_idx in range(0, total_samples, batch_size):
            opt.zero_grad()
            
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_states = train_states[batch_indices]
            batch_values = train_values[batch_indices]
            batch_points = train_points[batch_indices]
            
            _, value, value_error, points = net(batch_states, apply_softmax=False)
            
            with torch.no_grad():
                preds = torch.argmax(value, dim=1)
                all_preds[batch_indices] = preds
            
            val_probs = F.softmax(value, dim=1)
            v_pred = val_probs[:, 0] - val_probs[:, 2]
            
            v_true = torch.zeros_like(v_pred)
            v_true[batch_values == 0] = 1.0
            v_true[batch_values == 1] = 0.0
            v_true[batch_values == 2] = -1.0
            
            target_error = torch.abs(v_true - v_pred).unsqueeze(1).detach()
            
            loss_value = F.cross_entropy(value, batch_values, weight=class_weights)
            loss_points = F.cross_entropy(points, batch_points)
            loss_error = F.mse_loss(value_error, target_error)
            
            total_loss = loss_value + loss_points + loss_error
            total_loss.backward()
            opt.step()
            
        pred_counts = torch.bincount(all_preds, minlength=3)
        running_pred_wins = pred_counts[0].item()
        running_pred_draws = pred_counts[1].item()
        running_pred_losses = pred_counts[2].item()
        
        running_misc_wins = ((train_values == 0) & (all_preds != 0)).sum().item()
        running_misc_draws = ((train_values == 1) & (all_preds != 1)).sum().item()
        running_misc_losses = ((train_values == 2) & (all_preds != 2)).sum().item()
        
        total_misc = running_misc_wins + running_misc_draws + running_misc_losses
        print(f" MISCLASSIFIED: {(total_misc / total_samples * 100):.2f}%")
        print("-" * 62)
        print(f" Outcome Class | True Count | Model Predicted | Misclassified")
        print(f" --------------|------------|-----------------|--------------")
        print(f" Win  (0)      | {running_true_wins:<10} | {running_pred_wins:<15} | {running_misc_wins}")
        print(f" Draw (1)      | {running_true_draws:<10} | {running_pred_draws:<15} | {running_misc_draws}")
        print(f" Loss (2)      | {running_true_losses:<10} | {running_pred_losses:<15} | {running_misc_losses}")
        print("-" * 62)
        print(f"Iteration {iteration:03d} complete in {time.time() - t0:.4f} seconds")
        
        if prof is not None:
            prof.step()