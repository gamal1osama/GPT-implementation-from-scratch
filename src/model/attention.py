"""Attention mechanisms: single-head and multi-head self-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of causal (masked) self-attention."""

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # causal mask – not a learnable parameter
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Scaled dot-product attention scores
        scale = C ** -0.5
        wei = q @ k.transpose(-2, -1) * scale          # (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )                                               # (B, T, T)
        wei = F.softmax(wei, dim=-1)                    # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v                                   # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
