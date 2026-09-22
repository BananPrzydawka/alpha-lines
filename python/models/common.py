"""Configuration-independent building blocks for the convolutional models."""
from math import gcd

from torch import nn


def group_norm(channels, options):
    return nn.GroupNorm(gcd(options["groups"], channels), channels,
                        eps=options["eps"], affine=options["affine"])


def spatial_head(channels, hidden, norm, outputs=1):
    return nn.Sequential(
        nn.Conv2d(channels, hidden, 1, bias=False),
        group_norm(hidden, norm),
        nn.SiLU(),
        nn.Conv2d(hidden, outputs, 1),
    )


def immediate_score_head(channels, hidden, norm):
    """Predict both current scores from the shared residual tower."""
    return nn.Sequential(
        nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
        group_norm(hidden, norm),
        nn.SiLU(),
        nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
        group_norm(hidden, norm),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d((5, 8)),
        nn.Flatten(),
        nn.Linear(hidden * 5 * 8, 2 * 81),
    )


def prepare_inputs(board):
    if board.dim() == 3:
        board = board.unsqueeze(0)
    if board.dim() != 4 or board.shape[1:] != (5, 10, 16):
        raise ValueError("board must have shape (batch, 5, 10, 16) or (5, 10, 16)")
    return board
