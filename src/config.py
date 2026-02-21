"""
Configuration loader.

Reads config/config.yaml and exposes typed dataclasses for all settings.
"""

from __future__ import annotations

import yaml
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.2
    vocab_size: Optional[int] = None  # set dynamically after tokenisation


@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4


@dataclass
class DataConfig:
    raw_data_path: str = "data/raw/input.txt"
    processed_dir: str = "data/processed"
    train_split: float = 0.9


@dataclass
class PathsConfig:
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    device: str = "auto"
    seed: int = 1337

    def get_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load a YAML config file and return a typed Config object."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        device=raw.get("device", "auto"),
        seed=raw.get("seed", 1337),
    )
