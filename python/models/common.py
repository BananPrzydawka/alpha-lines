"""Configuration-independent building blocks for the convolutional models."""
from math import gcd

from torch import nn


def group_norm(channels, options):
    return nn.GroupNorm(gcd(options["groups"], channels), channels,
                        eps=options["eps"], affine=options["affine"])


def spatial_head(channels, hidden, norm):
    return nn.Sequential(
        nn.Conv2d(channels, hidden, 1, bias=False),
        group_norm(hidden, norm),
        nn.SiLU(),
        nn.Conv2d(hidden, 1, 1),
    )


def score_embedding(channels, hidden):
    return nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, channels))


def prepare_inputs(board, scores):
    if board.dim() == 3:
        board = board.unsqueeze(0)
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    if board.dim() != 4 or board.shape[1:] != (5, 10, 16):
        raise ValueError("board must have shape (batch, 5, 10, 16) or (5, 10, 16)")
    if scores.dim() != 2 or scores.shape != (board.shape[0], 2):
        raise ValueError("scores must have shape (batch, 2), ordered current player, opponent")
    return board, scores.to(device=board.device, dtype=board.dtype) / 80.0
