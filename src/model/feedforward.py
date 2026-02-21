"""Position-wise feed-forward network used inside each Transformer block."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP with ReLU activation and an inner dimension of 4 × n_embd."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
