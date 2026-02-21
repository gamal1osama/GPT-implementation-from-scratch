"""
GPT Language Model.

A character-level decoder-only Transformer (GPT-style) that learns to
predict the next token in a sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block


class GPTLanguageModel(nn.Module):
    """
    Decoder-only Transformer language model.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary.
    n_embd : int
        Embedding / model dimension.
    n_head : int
        Number of attention heads per block.
    n_layer : int
        Number of stacked Transformer blocks.
    block_size : int
        Maximum sequence length (context window).
    dropout : float
        Dropout probability applied in attention and FFN.
    device : str
        Device string ("cuda" or "cpu") – needed for positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
        device: str,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f   = nn.LayerNorm(n_embd)   # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialise weights (following GPT-2 paper)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        idx : (B, T) long tensor – token indices.
        targets : (B, T) long tensor – ground-truth next tokens (optional).

        Returns
        -------
        logits : (B, T, vocab_size)
        loss   : scalar cross-entropy loss, or None when targets is None.
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                           # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )                                                                    # (T, C)
        x = tok_emb + pos_emb                                               # (B, T, C)
        x = self.blocks(x)                                                  # (B, T, C)
        x = self.ln_f(x)                                                    # (B, T, C)
        logits = self.lm_head(x)                                            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Auto-regressively generate `max_new_tokens` tokens.

        Parameters
        ----------
        idx : (B, T) – current context.
        max_new_tokens : int – number of tokens to generate.

        Returns
        -------
        (B, T + max_new_tokens) tensor.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]          # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                     # (B, vocab_size)
            probs  = F.softmax(logits, dim=-1)            # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)       # (B, T+1)
        return idx

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
