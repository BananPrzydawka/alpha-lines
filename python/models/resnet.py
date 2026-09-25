"""Historical squeeze-and-excitation ResNet with the current seven heads."""

from torch import nn
from torch.nn import functional as F

from config import settings
from models.common import group_norm, spatial_head, categorical_score_head, prepare_inputs


class SqueezeExcite(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False), nn.SiLU(),
            nn.Linear(bottleneck, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, features):
        batch, channels, _, _ = features.shape
        weights = self.excite(self.squeeze(features).reshape(batch, channels))
        return features * weights.reshape(batch, channels, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, bottleneck, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = group_norm(channels, norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = group_norm(channels, norm)
        self.se = SqueezeExcite(channels, bottleneck)

    def forward(self, features):
        residual = features
        features = F.silu(self.norm1(self.conv1(features)))
        features = self.se(self.norm2(self.conv2(features)))
        return F.silu(features + residual)


class ResNet(nn.Module):
    """Five board planes to policy, value, mark, score, and opponent heads."""

    def __init__(self, options=None):
        super().__init__()
        options = settings['resnet_model'] if options is None else options
        channels = options['filters']
        norm = options['group_norm']
        self.conv_input = nn.Conv2d(5, channels, 3, padding=1, bias=False)
        self.norm_input = group_norm(channels, norm)
        self.tower = nn.Sequential(*(
            ResidualBlock(channels, options['se_hidden'], norm)
            for _ in range(options['blocks'])
        ))
        self.policy_head = spatial_head(channels, options['policy_filters'], norm)
        self.opponent_policy_head = spatial_head(channels, options['opponent_policy_filters'], norm)
        self.action_value_head = spatial_head(channels, options['action_value_filters'], norm)
        self.mark_class_head = spatial_head(channels, options.get('mark_class_filters',
                                             options['action_value_filters']), norm, 6)
        self.discounted_mark_head = spatial_head(channels, options.get('discounted_mark_filters',
                                                  options['action_value_filters']), norm, 8)
        self.immediate_score_head = categorical_score_head(channels,
                                                options.get('immediate_score_filters', 32), norm)
        self.discounted_score_head = categorical_score_head(channels,
                                                 options.get('discounted_score_filters', 32), norm)

    def forward(self, board):
        board = prepare_inputs(board)
        features = self.tower(F.silu(self.norm_input(self.conv_input(board))))
        return (self.policy_head(features).squeeze(1),
                self.action_value_head(features).squeeze(1),
                self.mark_class_head(features),
                self.immediate_score_head(features).reshape(-1, 2, 81),
                self.discounted_score_head(features).reshape(-1, 2, 81),
                self.discounted_mark_head(features),
                self.opponent_policy_head(features).squeeze(1))
