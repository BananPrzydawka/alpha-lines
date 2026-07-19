import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (height, width, board_size, filters, bottleneck, resblock_number, policy_filters, value_fc, error_fc, point_fc)

class se_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(filters, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, filters, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class res_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.se = se_block()
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x
        out = self.silu(self.bn1(self.conv1(x)))
        # out = self.silu(self.conv1(x))
        out = self.bn2(self.conv2(out))
        # out = (self.conv2(out))
        out = self.se(out)
        out += residual
        return self.silu(out)

class alpha_lines_net(nn.Module):
    """encoded game (batch, 7, height, width) => policy logits, value logits, value error scalar, points logits"""
    def __init__(self):
        super().__init__()
        
        self.conv_input = nn.Conv2d(7, filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(filters)
        self.tower = nn.Sequential(*[res_block() for _ in range(resblock_number)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, policy_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_filters),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(policy_filters * board_size, board_size), 
            nn.Unflatten(1, (height, width))
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(bottleneck * board_size, value_fc),
            nn.SiLU(),
            nn.Linear(value_fc, 3) # Win, Draw, Loss
        )

        self.error_head = nn.Sequential(
            nn.Conv2d(filters, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(bottleneck * board_size, error_fc),
            nn.SiLU(),
            nn.Linear(error_fc, 1)
        )

        self.point_head = nn.Sequential(
            nn.Conv2d(filters, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(bottleneck * board_size, point_fc),
            nn.SiLU(),
            nn.Linear(point_fc, (board_size+1))
        )

    def forward(self, x, apply_softmax=False):
        # ecpects input of dim 4, but in case an idiot calls it
        if x.dim() == 3:
            print("dumbass")
            x = x.unsqueeze(0)

        s = F.silu(self.bn_input(self.conv_input(x)))
        # s = F.silu(self.conv_input(x))
        s = self.tower(s)
        
        policy       = self.policy_head(s) # Logits
        value        = self.value_head(s)  # Logits
        value_error  = self.error_head(s)  # Scalar
        points       = self.point_head(s)  # Logits
        
        if apply_softmax == True:
            return (
                F.softmax(policy.flatten(1), dim=1).view_as(policy), 
                F.softmax(value, dim=1), 
                value_error, 
                F.softmax(points, dim=1)
            )

        return policy, value, value_error, points

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path, device=None):
        self.load_state_dict(torch.load(path, map_location=device))


# from torchinfo import summary
# model = alpha_lines_net()
# model = model.to(torch.bfloat16)
# summary(model, input_size=(1, 7, 10, 16), dtypes=[torch.bfloat16])