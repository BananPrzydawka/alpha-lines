"""
Training loop for Alpha-Lines
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import AlphaLinesNet
from config import (
    LEARNING_RATE, WEIGHT_DECAY, MOMENTUM,
    MINIBATCH_SIZE, NUM_EPOCHS_PER_BATCH,
    MAX_POINT_DIFFERENCE, DEVICE
)


class Trainer:
    """
    Trainer for the Alpha-Lines model.
    """
    
    def __init__(self, model=None, device=None):
        if device is None:
            device = DEVICE if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        if model is None:
            self.model = AlphaLinesNet().to(device)
        else:
            self.model = model.to(device)
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
        
        # Loss functions
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()
        self.point_loss_fn = nn.CrossEntropyLoss()
    
    def train_batch(self, states, policies, values, point_diff_classes):
        """
        Train the model on a batch of data.
        
        Args:
            states: (N, 6, 10, 16) numpy array
            policies: (N, 10, 16) numpy array (probability distributions)
            values: (N,) numpy array (-1 to 1)
            point_diff_classes: (N,) numpy array (int labels 0-160)
        
        Returns:
            dict of loss values
        """
        self.model.train()
        
        # Convert to tensors
        states_tensor = torch.from_numpy(states).float().to(self.device)
        policies_tensor = torch.from_numpy(policies).float().to(self.device)
        values_tensor = torch.from_numpy(values).float().to(self.device)
        point_diff_tensor = torch.from_numpy(point_diff_classes).long().to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_point_loss = 0
        num_batches = 0
        
        # Train for multiple epochs
        for epoch in range(NUM_EPOCHS_PER_BATCH):
            # Shuffle data
            indices = np.random.permutation(len(states))
            
            # Mini-batch training
            for start_idx in range(0, len(states), MINIBATCH_SIZE):
                end_idx = min(start_idx + MINIBATCH_SIZE, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_policies = policies_tensor[batch_indices]
                batch_values = values_tensor[batch_indices]
                batch_point_diff = point_diff_tensor[batch_indices]

                # Forward pass
                policy_pred, value_pred, point_pred = self.model(batch_states)
                
                # Policy loss - KL divergence
                # Add small epsilon to avoid log(0)
                policy_pred_log = torch.log(policy_pred + 1e-8)
                policy_loss = self.policy_loss_fn(policy_pred_log, batch_policies)
                
                # Value loss - MSE
                value_loss = self.value_loss_fn(value_pred, batch_values)
                print(value_pred[30].item(), batch_values[30].item())
                
                # Point difference loss - cross entropy
                point_loss = self.point_loss_fn(point_pred, batch_point_diff)
                
                # Total loss
                loss = policy_loss + value_loss + point_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_point_loss += point_loss.item()
                num_batches += 1
        
        # Average losses
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        avg_point_loss = total_point_loss / max(num_batches, 1)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'point_loss': avg_point_loss,
            'total_loss': avg_policy_loss + avg_value_loss + avg_point_loss
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        self.model.save_checkpoint(path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        self.model.load_checkpoint(path, device=self.device)