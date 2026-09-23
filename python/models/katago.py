"""KataGo-inspired nested bottlenecks with spatial action and score heads.

Uses our GroupNorm, SiLU, and outer squeeze-excitation, not a full reproduction
of KataGo's normalization, global pooling, heads, or training recipe.
"""
from torch import nn
from torch.nn import functional as F

from config import settings
from models.common import group_norm, spatial_head, categorical_score_head, prepare_inputs


class InnerResidualBlock(nn.Module):
    """Two spatial convolutions at half the outer channel width."""
    def __init__(self, channels, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = group_norm(channels, norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = group_norm(channels, norm)

    def forward(self, features):
        residual = features
        features = F.silu(self.norm1(self.conv1(features)))
        features = self.norm2(self.conv2(features))
        return F.silu(residual + features)


class NestedBottleneck(nn.Module):
    """C -> C/2 -> two inner residual blocks -> C -> outer residual add."""
    def __init__(self, channels, se_hidden, norm):
        super().__init__()
        inner_channels = channels // 2
        self.reduce = nn.Conv2d(channels, inner_channels, 1, bias=False)
        self.reduce_norm = group_norm(inner_channels, norm)
        self.inner_blocks = nn.Sequential(
            InnerResidualBlock(inner_channels, norm),
            InnerResidualBlock(inner_channels, norm),
        )
        self.expand = nn.Conv2d(inner_channels, channels, 1, bias=False)
        self.expand_norm = group_norm(channels, norm)
        # Match the existing ResNet's SE on the outer residual branch.
        self.excite = nn.Sequential(
            nn.Linear(channels, se_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(se_hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features):
        residual = features
        features = F.silu(self.reduce_norm(self.reduce(features)))
        features = self.inner_blocks(features)
        features = self.expand_norm(self.expand(features))
        channel_weights = self.excite(features.mean(dim=(2, 3)))
        features = features * channel_weights[:, :, None, None]
        return F.silu(residual + features)


class KataGoNet(nn.Module):
    """forward(board) -> policy, values, mark/score heads, and opponent policy.

    Score logits are ordered current player then opponent. Outputs are unbounded.
    """
    def __init__(self, mark_classes=True):
        super().__init__()
        options = settings["katago_model"]
        channels = options["filters"]
        if channels < 2 or channels % 2:
            raise ValueError("katago_model.filters must be a positive even integer")
        norm = options["group_norm"]

        self.conv_input = nn.Conv2d(5, channels, 3, padding=1, bias=False)
        self.norm_input = group_norm(channels, norm)

        self.tower = nn.Sequential(*(
            NestedBottleneck(channels, options["se_hidden"], norm)
            for _ in range(options["blocks"])
        ))
        self.policy_head = spatial_head(channels, options["policy_filters"], norm)
        self.opponent_policy_head = spatial_head(channels, options["opponent_policy_filters"], norm)
        self.action_value_head = spatial_head(channels, options["action_value_filters"], norm)
        self.mark_class_head = (spatial_head(channels, options.get("mark_class_filters", options["action_value_filters"]), norm, 6)
                                if mark_classes else None)
        self.discounted_mark_head = (spatial_head(channels, options.get("discounted_mark_filters", options["action_value_filters"]), norm, 8)
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
                self.discounted_score_head(features).reshape(-1, 2, 81),
                self.discounted_mark_head(features) if self.discounted_mark_head is not None else None,
                self.opponent_policy_head(features).squeeze(1))
