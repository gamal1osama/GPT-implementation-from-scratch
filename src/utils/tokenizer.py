"""
Character-level tokenizer.

Maps every unique character in a corpus to an integer and back.
The vocabulary is derived from the training corpus and saved to disk so that
inference can reuse the exact same mapping.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Simple character-level tokenizer (no sub-word splitting)."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size: int = len(chars)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer token ids."""
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token ids back to a string."""
        return "".join(self.itos[i] for i in ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialise the tokenizer to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            # JSON keys must be strings; store int keys as strings
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a previously saved tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        obj = cls.__new__(cls)
        obj.vocab_size = payload["vocab_size"]
        obj.stoi = payload["stoi"]
        obj.itos = {int(k): v for k, v in payload["itos"].items()}
        return obj

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
