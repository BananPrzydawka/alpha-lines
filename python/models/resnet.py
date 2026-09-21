"""Score-conditioned ResNet with spatial policy logits and action values."""
import torch.nn as nn
import torch.nn.functional as F

from config import settings
from models.common import group_norm, spatial_head, score_embedding, prepare_inputs


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
    """forward(board, scores) -> (policy_logits, action_values), each (B, 10, 16).

    Scores are raw values in [0, 80], ordered current player then opponent;
    board player planes must use the same perspective. Outputs are unbounded.
    """
    def __init__(self):
        super().__init__()
        options = settings["resnet_model"]
        channels = options["filters"]
        norm = options["group_norm"]
        self.conv_input = nn.Conv2d(5, channels, 3, padding=1, bias=False)
        self.norm_input = group_norm(channels, norm)
        self.score_embed = score_embedding(channels, options["score_embed_hidden"])
        self.tower = nn.Sequential(*(
            res_block(channels, options["se_hidden"], norm)
            for _ in range(options["blocks"])
        ))
        self.policy_head = spatial_head(channels, options["policy_filters"], norm)
        self.action_value_head = spatial_head(channels, options["action_value_filters"], norm)

    def forward(self, board, scores):
        board, scores = prepare_inputs(board, scores)
        features = self.norm_input(self.conv_input(board))
        features = F.silu(features + self.score_embed(scores)[:, :, None, None])
        features = self.tower(features)
        return (self.policy_head(features).squeeze(1),
                self.action_value_head(features).squeeze(1))
