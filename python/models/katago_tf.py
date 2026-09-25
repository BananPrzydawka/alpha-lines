"""KataGo tower with both inner residual blocks replaced by attention blocks."""

import torch
from torch import nn
from torch.nn import functional as F

from config import settings
from models.katago import KataGoNet


class SpatialTransformerBlock(nn.Module):
    """Pre-norm self-attention and MLP over all 160 board locations."""

    def __init__(self, channels, heads, mlp_ratio):
        super().__init__()
        if channels % heads:
            raise ValueError('attention_heads must divide the inner channel count')
        self.heads = heads
        self.position = nn.Parameter(torch.empty(1, 160, channels))
        nn.init.normal_(self.position, std=0.02)
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3*channels)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_ratio*channels), nn.GELU(),
            nn.Linear(mlp_ratio*channels, channels),
        )

    def forward(self, features):
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens + self.position)
        qkv = self.qkv(normalized).reshape(batch, height*width, 3,
                                            self.heads, channels//self.heads)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(queries, keys, values)
        attended = attended.transpose(1, 2).reshape(batch, height*width, channels)
        tokens = tokens + self.projection(attended)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class KataGoTFNet(KataGoNet):
    def __init__(self, options=None):
        options = settings['katago_tf_model'] if options is None else options
        heads = options['attention_heads']
        ratio = options['mlp_ratio']
        super().__init__(options, inner_block_factory=lambda channels, norm:
                         SpatialTransformerBlock(channels, heads, ratio))
