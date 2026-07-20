"""
Neural network model for Alpha-Lines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    BOARD_HEIGHT, BOARD_WIDTH,
    NUM_RESIDUAL_BLOCKS, NUM_FILTERS,
    POLICY_HEAD_FILTERS, VALUE_HEAD_FILTERS, POINT_HEAD_FILTERS,
    MAX_POINT_DIFFERENCE
)


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and batch norm"""
    
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class AlphaLinesNet(nn.Module):
    """
    Neural network for Alpha-Lines.
    
    Input: 6 x 10 x 16 board encoding
    Output: 
        - policy: 10 x 16 probability distribution over moves (via conv)
        - value: scalar value estimate (-1 to 1)
        - point_diff: 161-way classification of point difference
    """
    
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(6, NUM_FILTERS, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(NUM_FILTERS)
        
        # Residual tower
        self.residual_tower = nn.ModuleList([
            ResidualBlock(NUM_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)
        ])
        
        # Policy head - convolutional, outputs 10x16x1
        self.policy_conv = nn.Conv2d(NUM_FILTERS, POLICY_HEAD_FILTERS, kernel_size=3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_FILTERS)
        self.policy_out = nn.Conv2d(POLICY_HEAD_FILTERS, 1, kernel_size=1, bias=False)
        
        # Value head
        self.value_conv = nn.Conv2d(NUM_FILTERS, VALUE_HEAD_FILTERS, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_FILTERS)
        value_flatten_size = VALUE_HEAD_FILTERS * BOARD_HEIGHT * BOARD_WIDTH
        self.value_fc1 = nn.Linear(value_flatten_size, 256)
        self.value_out = nn.Linear(256, 1)
        
        # Point predictor head
        self.point_conv = nn.Conv2d(NUM_FILTERS, POINT_HEAD_FILTERS, kernel_size=1, bias=False)
        self.point_bn = nn.BatchNorm2d(POINT_HEAD_FILTERS)
        point_flatten_size = POINT_HEAD_FILTERS * BOARD_HEIGHT * BOARD_WIDTH
        self.point_fc1 = nn.Linear(point_flatten_size, 256)
        self.point_out = nn.Linear(256, MAX_POINT_DIFFERENCE * 2 + 1)  # -80 to +80 = 161 classes
    
    def forward(self, x):
        """
        x: (batch, 6, 10, 16) board encoding
        Returns: policy (batch, 10, 16), value (batch,), point_diff (batch, 161)
        """
        # Initial conv
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.residual_tower:
            out = block(out)
        
        # Policy head - pure convolutional
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = self.policy_out(policy)  # (batch, 1, 10, 16)
        policy = policy.squeeze(1)  # (batch, 10, 16)
        # Softmax over the board (flatten, softmax, reshape)
        batch_size = policy.shape[0]
        policy_flat = policy.view(batch_size, -1)
        policy = F.softmax(policy_flat, dim=-1).view(batch_size, BOARD_HEIGHT, BOARD_WIDTH)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_out(value)).squeeze(-1)
        
        # Point predictor head
        point = F.relu(self.point_bn(self.point_conv(out)))
        point = point.view(batch_size, -1)
        point = F.relu(self.point_fc1(point))
        point = self.point_out(point)  # (batch, 161)
        # No softmax here - we'll use CrossEntropyLoss which includes it
        
        return policy, value, point
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)
    
    def load_checkpoint(self, path, device='cpu'):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])