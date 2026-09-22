"""ResNet with shared spatial policy, action-value, mark and score heads."""
import torch.nn as nn
import torch.nn.functional as F

from config import settings
from models.common import group_norm, spatial_head, categorical_score_head, prepare_inputs


class se_block(nn.Module):
    def __init__(self, filters, bottleneck):
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
    def __init__(self, filters, bottleneck, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.norm1 = group_norm(filters, norm)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.norm2 = group_norm(filters, norm)
        self.se = se_block(filters, bottleneck)
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x
        out = self.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.silu(out)

class ResNet(nn.Module):
    """forward(board) -> policy, action values, mark classes, current and discounted scores.

    Score logits are ordered current player then opponent. Outputs are unbounded.
    """
    def __init__(self, mark_classes=True):
        super().__init__()
        options = settings["resnet_model"]
        channels = options["filters"]
        norm = options["group_norm"]
        self.conv_input = nn.Conv2d(5, channels, 3, padding=1, bias=False)
        self.norm_input = group_norm(channels, norm)
        self.tower = nn.Sequential(*(
            res_block(channels, options["se_hidden"], norm)
            for _ in range(options["blocks"])
        ))
        self.policy_head = spatial_head(channels, options["policy_filters"], norm)
        self.action_value_head = spatial_head(channels, options["action_value_filters"], norm)
        self.mark_class_head = (spatial_head(channels, options.get("mark_class_filters", options["action_value_filters"]), norm, 6)
                                if mark_classes else None)
        self.immediate_score_head = categorical_score_head(channels, options.get("immediate_score_filters", 32), norm)
        self.discounted_score_head = categorical_score_head(channels, options.get("discounted_score_filters", 32), norm)

    def forward(self, board):
        board = prepare_inputs(board)
        features = F.silu(self.norm_input(self.conv_input(board)))
        features = self.tower(features)
        return (self.policy_head(features).squeeze(1),
                self.action_value_head(features).squeeze(1),
                self.mark_class_head(features) if self.mark_class_head is not None else None,
                self.immediate_score_head(features).reshape(-1, 2, 81),
                self.discounted_score_head(features).reshape(-1, 2, 81))
