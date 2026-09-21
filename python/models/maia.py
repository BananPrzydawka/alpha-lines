"""Maia-3 small-model adaptation: square tokens and pooled geometric attention bias.

Configurable pre-LayerNorm encoder blocks with pooled GAB.
No history, rating embeddings, or dropout. GAB templates are shared across
heads and layers; each layer generates its own board-dependent coefficients.
"""
import torch
from torch import nn
from torch.nn import functional as F

from config import height, width, settings


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, gab_dim):
        super().__init__()
        self.heads = heads
        self.gab_dim = gab_dim
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.gab = nn.Sequential(
            nn.Linear(dim, gab_dim),
            nn.GELU(),
            nn.LayerNorm(gab_dim),
            nn.Linear(gab_dim, heads * gab_dim),
            nn.GELU(),
            nn.LayerNorm(heads * gab_dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, tokens, templates):
        batch_size, square_count, embedding_dim = tokens.shape
        head_dim = embedding_dim // self.heads
        normalized = self.norm1(tokens)

        # Split each square's embedding into queries, keys, and values per head.
        # Each resulting tensor has shape (batch, heads, squares, head_dim).
        qkv = self.qkv(normalized)
        qkv = qkv.reshape(batch_size, square_count, 3, self.heads, head_dim)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # Summarize this board, then mix the shared geometric bias templates.
        board_summary = normalized.mean(dim=1)
        coefficients = self.gab(board_summary)
        coefficients = coefficients.reshape(batch_size, self.heads, self.gab_dim)
        attention_bias = F.linear(coefficients, templates)
        attention_bias = attention_bias.reshape(
            batch_size, self.heads, square_count, square_count
        )

        attended = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attention_bias.to(queries.dtype)
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, square_count, embedding_dim
        )
        tokens = tokens + self.projection(attended)
        return tokens + self.mlp(self.norm2(tokens))


class MaiaNet(nn.Module):
    """Five board planes -> both players' score logits (batch, 2, 81).

    Policy and value heads are kept commented out for future experiments.
    """
    def __init__(self):
        super().__init__()
        options = settings["maia_model"]
        dim = options["dim"]
        depth = options["layers"]
        head_dim = options["head_dim"]
        mlp_ratio = options["mlp_ratio"]
        gab_dim = options["gab_dim"]
        dimensions = (dim, depth, head_dim, mlp_ratio, gab_dim)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("maia_model dimensions must be positive integers")
        if dim % head_dim:
            raise ValueError("maia_model.head_dim must divide dim")
        squares = torch.arange(80)
        # Eight playable squares per row, alternating even and odd columns.
        rows = squares // 8
        columns = 2 * (squares % 8) + rows % 2
        indices = rows * width + columns
        self.register_buffer("indices", indices, persistent=False)
        self.embedding = nn.Linear(5, dim)
        self.templates = nn.Parameter(torch.empty(80 * 80, gab_dim))
        nn.init.normal_(self.templates, std=gab_dim ** -0.5)
        self.layers = nn.ModuleList([
            EncoderBlock(dim, dim // head_dim, mlp_ratio, gab_dim) for _ in range(depth)
        ])
        score_options = options["score_head"]
        square_features = score_options["square_features"]
        hidden_dim = score_options["hidden_dim"]
        if any(type(value) is not int or value < 1
               for value in (square_features, hidden_dim)):
            raise ValueError("maia_model.score_head dimensions must be positive integers")

        # Keep squares separate so the score head can combine features by location.
        self.score_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, square_features),
            nn.GELU(),
            nn.Flatten(start_dim=1),
            nn.Linear(len(squares) * square_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * 81),
        )
        # self.policy_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, 1),
        # )
        # self.value_head = self.pooled_head(dim, 3)

    @staticmethod
    def pooled_head(dim, outputs):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, outputs),
        )

    def forward(self, board, apply_softmax=False):
        if board.dim() == 3:
            board = board.unsqueeze(0)
        batch_size = board.shape[0]

        # (batch, planes, height, width) -> (batch, 80 playable squares, planes).
        squares = board.flatten(start_dim=2)
        playable_squares = squares.index_select(2, self.indices)
        tokens = self.embedding(playable_squares.transpose(1, 2))
        for layer in self.layers:
            tokens = layer(tokens, self.templates)

        scores = self.score_head(tokens).reshape(batch_size, 2, 81)
        return scores.softmax(-1) if apply_softmax else scores

        # To enable the other heads, initialize them above and return them here.
        # square_logits = self.policy_head(tokens).squeeze(-1)
        # policy = square_logits.new_full((batch_size, height * width), float("-inf"))
        # policy = policy.scatter(1, self.indices.expand(batch_size, -1), square_logits)
        # policy = policy.reshape(batch_size, height, width)
        # value = self.value_head(tokens.mean(dim=1))
        # return policy, value, scores
