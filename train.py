"""
train.py – entry point for training the GPT language model.

Usage
-----
    python train.py
    python train.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make sure the project root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.data.dataset import TextDataset
from src.model.gpt import GPTLanguageModel
from src.training.trainer import Trainer
from src.utils.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GPT language model on a text corpus."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML configuration file (default: config/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration 
    config = load_config(args.config)
    device = config.get_device()
    print(f"Device : {device}")
    print(f"Config : {args.config}\n")

    # Reproducibility 
    torch.manual_seed(config.seed)

    # Tokeniser
    data_path = Path(config.data.raw_data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at '{data_path}'.\n"
            "Download it with:\n"
            "  wget https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt -O data/raw/input.txt"
        )

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size : {tokenizer.vocab_size} unique characters\n")

    # Save tokenizer so generate.py can reuse the same mapping
    tokenizer_path = Path(config.data.processed_dir) / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved → {tokenizer_path}")

    #  Dataset 
    dataset = TextDataset(data_path, tokenizer, config.data.train_split)

    #  Model 
    model = GPTLanguageModel(
        vocab_size  = config.model.vocab_size,
        n_embd      = config.model.n_embd,
        n_head      = config.model.n_head,
        n_layer     = config.model.n_layer,
        block_size  = config.model.block_size,
        dropout     = config.model.dropout,
        device      = device,
    ).to(device)

    print(f"Parameters      : {model.num_parameters() / 1e6:.2f}M\n")

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate
    )

    # Train
    trainer = Trainer(model, optimizer, dataset, config, device)
    trainer.train()

    # Save final checkpoint
    final_ckpt = Path(config.paths.checkpoints_dir) / "final_model.pt"
    trainer.save_checkpoint(str(final_ckpt))
    print(f"\nFinal checkpoint saved → {final_ckpt}")

    # Quick sample
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=200)
    print("\n── Sample output")
    print(tokenizer.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
