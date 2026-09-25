"""Maia-style square-token encoder adapted to the KLENT seven-head interface."""

import torch
from torch import nn
from torch.nn import functional as F

from config import settings
from models.common import prepare_inputs


class EncoderBlock(nn.Module):
    """Historical pre-LayerNorm attention with pooled geometric bias."""

    def __init__(self, dim, heads, mlp_ratio, gab_dim, big_version=False):
        super().__init__()
        self.heads = heads
        self.gab_dim = gab_dim
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim)
        self.projection = nn.Linear(dim, dim)
        # Large Chessformer GAB: d1=32 per square, then flatten all 80 squares.
        # Small GAB uses average pooling and skips this projection entirely.
        self.gab_input = nn.Linear(dim,32) if big_version else None
        self.gab = nn.Sequential(
            nn.Linear(80*32 if big_version else dim, gab_dim),
            nn.GELU(), nn.LayerNorm(gab_dim),
            nn.Linear(gab_dim, heads*gab_dim), nn.GELU(), nn.LayerNorm(heads*gab_dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio), nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim),
        )

    def forward(self, tokens, templates):
        batch, squares, dim = tokens.shape
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized).reshape(batch, squares, 3, self.heads, dim//self.heads)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        summary = (self.gab_input(normalized).flatten(1) if self.gab_input is not None
                   else normalized.mean(dim=1))
        coefficients = self.gab(summary).reshape(batch,self.heads,self.gab_dim)
        bias = F.linear(coefficients, templates).reshape(batch,self.heads,squares,squares)
        attended = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=bias.to(queries.dtype))
        attended = attended.transpose(1, 2).reshape(batch,squares,dim)
        tokens = tokens + self.projection(attended)
        return tokens + self.mlp(self.norm2(tokens))


class MaiaNet(nn.Module):
    """Consume only 80 playable squares and return full-grid KLENT logits."""

    def __init__(self, options=None):
        super().__init__()
        options = settings['maia_model'] if options is None else options
        dim, depth = options['dim'], options['layers']
        big_version = options.get('maia_big_version',False)
        if type(big_version) is not bool:
            raise ValueError('maia_model.maia_big_version must be a boolean')
        head_dim, mlp_ratio, gab_dim = (options[key] for key in
                                        ('head_dim','mlp_ratio','gab_dim'))
        if any(type(value) is not int or value < 1 for value in
               (dim,depth,head_dim,mlp_ratio,gab_dim)) or dim % head_dim:
            raise ValueError('maia_model dimensions must be positive and head_dim must divide dim')
        squares = torch.arange(80)
        rows = squares//8
        columns = 2*(squares%8) + rows%2
        self.register_buffer('indices', rows*16+columns, persistent=False)
        self.embedding = nn.Linear(5,dim)
        self.templates = nn.Parameter(torch.empty(80*80,gab_dim))
        nn.init.normal_(self.templates,std=gab_dim**-0.5)
        self.layers = nn.ModuleList([
            EncoderBlock(dim,dim//head_dim,mlp_ratio,gab_dim,big_version)
            for _ in range(depth)
        ])
        self.policy_head = self.square_head(dim,1)
        self.action_value_head = self.square_head(dim,1)
        self.mark_class_head = self.square_head(dim,6)
        self.discounted_mark_head = self.square_head(dim,8)
        self.opponent_policy_head = self.square_head(dim,1)
        self.immediate_score_head = self.score_head(dim,options['score_head'])
        self.discounted_score_head = self.score_head(dim,options['score_head'])

    @staticmethod
    def square_head(dim, outputs):
        return nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,outputs))

    @staticmethod
    def score_head(dim, options):
        square_features, hidden = options['square_features'], options['hidden_dim']
        if any(type(value) is not int or value < 1 for value in (square_features,hidden)):
            raise ValueError('maia_model.score_head dimensions must be positive integers')
        return nn.Sequential(
            nn.LayerNorm(dim),nn.Linear(dim,square_features),nn.GELU(),
            nn.Flatten(start_dim=1),nn.Linear(80*square_features,hidden),nn.GELU(),
            nn.Linear(hidden,2*81),
        )

    def expand_spatial(self, logits):
        batch, _, channels = logits.shape
        full = logits.new_zeros((batch,160,channels))
        indices = self.indices.reshape(1,80,1).expand(batch,80,channels)
        return full.scatter(1,indices,logits).transpose(1,2).reshape(batch,channels,10,16)

    def forward(self, board):
        board = prepare_inputs(board)
        batch = board.shape[0]
        playable = board.flatten(2).index_select(2,self.indices)
        tokens = self.embedding(playable.transpose(1,2))
        for layer in self.layers:
            tokens = layer(tokens,self.templates)
        return (self.expand_spatial(self.policy_head(tokens)).squeeze(1),
                self.expand_spatial(self.action_value_head(tokens)).squeeze(1),
                self.expand_spatial(self.mark_class_head(tokens)),
                self.immediate_score_head(tokens).reshape(batch,2,81),
                self.discounted_score_head(tokens).reshape(batch,2,81),
                self.expand_spatial(self.discounted_mark_head(tokens)),
                self.expand_spatial(self.opponent_policy_head(tokens)).squeeze(1))
