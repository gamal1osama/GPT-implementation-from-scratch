"""
Dataset utilities.

Wraps the raw text file, tokenises it, and provides a `get_batch` helper
that returns random batches of (input, target) tensors for training or
validation.
"""

from __future__ import annotations

import torch
from pathlib import Path

from src.utils.tokenizer import CharTokenizer


class TextDataset:
    """
    Character-level text dataset.

    Parameters
    ----------
    data_path : str | Path
        Path to the plain-text file used for training.
    tokenizer : CharTokenizer
        A fitted tokenizer; the dataset stores encoded tensors.
    train_split : float
        Fraction of tokens reserved for training (the rest go to validation).
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: CharTokenizer,
        train_split: float = 0.9,
    ) -> None:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(train_split * len(data))
        self.train_data: torch.Tensor = data[:n]
        self.val_data:   torch.Tensor = data[n:]

        print(
            f"Dataset loaded – "
            f"train tokens: {len(self.train_data):,}  |  "
            f"val tokens:   {len(self.val_data):,}"
        )

    # ------------------------------------------------------------------
    # Batch sampling
    # ------------------------------------------------------------------
    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a random batch of (inputs, targets).

        Parameters
        ----------
        split : "train" | "val"
        batch_size : int
        block_size : int – sequence length.
        device : str

        Returns
        -------
        x : (batch_size, block_size) – input token ids.
        y : (batch_size, block_size) – target token ids (shifted by 1).
        """
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)
