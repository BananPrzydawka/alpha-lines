import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (height, width, board_size, filters, bottleneck, 
                    resblock_number, policy_filters, value_fc, error_fc, point_fc)

class alpha_lines_net(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep standard parameters to maintain initialization compatibility in main.py
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x, apply_softmax=False):
        # Expects batch dimensions: (B, 7, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        device = x.device

        # 1. Construct Mock Policy: 99% mass on top row, 1% on the rest
        # Create raw logit plane matching (H, W)
        mock_policy = torch.zeros((height, width), dtype=torch.float32, device=device)
        
        # Assign high logit values to the top row (row 0), baseline to others
        mock_policy[0, :] = 50.0
        mock_policy[1:, :] = 1.0
        
        # Replicate across batch dimension -> shape: (B, H, W)
        policy_logits = mock_policy.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Construct Mock Value: Balanced flat prediction (Draw-leaning evaluation)
        # Shape: (B, 3) representing Win, Draw, Loss
        value_logits = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
        value_logits[:, 1] = 5.0  # Elevate Draw index logit slightly

        # 3. Construct Auxiliary Heads: Flat scalars/logits
        value_error = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        points_logits = torch.zeros((batch_size, board_size + 1), dtype=torch.float32, device=device)

        if apply_softmax:
            # Flatten policy for standard categorical evaluation
            flat_policy = F.softmax(policy_logits.flatten(1), dim=1)
            
            # Recompute accurate distribution ratios if raw softmax squashes the 1% tail too low
            # Top row contains 'width' elements. Let top row elements share 0.99, others share 0.01
            top_row_count = width
            other_count = board_size - width
            
            prob_mask = torch.zeros(board_size, dtype=torch.float32, device=device)
            prob_mask[:top_row_count] = 0.99 / top_row_count
            prob_mask[top_row_count:] = 0.01 / other_count if other_count > 0 else 0.0
            
            policy_out = prob_mask.unsqueeze(0).expand(batch_size, -1).view_as(policy_logits)
            value_out = F.softmax(value_logits, dim=1)
            points_out = F.softmax(points_logits, dim=1)
            
            return policy_out, value_out, value_error, points_out

        return policy_logits, value_logits, value_error, points_logits

    def save_checkpoint(self, path):
        pass

    def load_checkpoint(self, path, device=None):
        pass