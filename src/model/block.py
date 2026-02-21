"""Transformer block: communication (attention) followed by computation (FFN)."""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class Block(nn.Module):
    """
    One Transformer decoder block.

    Architecture (pre-LN):
        x  →  LayerNorm  →  MultiHeadAttention  →  residual add
           →  LayerNorm  →  FeedForward          →  residual add
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))    # self-attention with residual
        x = x + self.ffwd(self.ln2(x))  # feed-forward with residual
        return x
