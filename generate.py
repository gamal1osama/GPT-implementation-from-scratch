"""
generate.py – generate text with a trained GPT checkpoint.

Usage
-----
    python generate.py
    python generate.py --checkpoint checkpoints/final_model.pt --tokens 500
    python generate.py --prompt "To be or not to be" --tokens 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.model.gpt import GPTLanguageModel
from src.utils.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with a trained GPT language model."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/final_model.pt",
        help="Path to a saved model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        default="data/processed/tokenizer.json",
        help="Path to the saved tokenizer JSON.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=500,
        help="Number of new tokens to generate (default: 500).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Seed text for generation. Leave empty to start from a zero token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Config & device
    config = load_config(args.config)
    device = config.get_device()

    # Tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at '{tokenizer_path}'.\n"
            "Run train.py first to generate the tokenizer."
        )
    tokenizer = CharTokenizer.load(tokenizer_path)
    config.model.vocab_size = tokenizer.vocab_size

    # Model
    model = GPTLanguageModel(
        vocab_size  = config.model.vocab_size,
        n_embd      = config.model.n_embd,
        n_head      = config.model.n_head,
        n_layer     = config.model.n_layer,
        block_size  = config.model.block_size,
        dropout     = config.model.dropout,
        device      = device,
    ).to(device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'.\n"
            "Run train.py first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}  (step={ckpt.get('step', '?')})\n")

    # Prepare context
    if args.prompt:
        context = torch.tensor(
            tokenizer.encode(args.prompt), dtype=torch.long, device=device
        ).unsqueeze(0)  # (1, T)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=args.tokens)

    generated_text = tokenizer.decode(output[0].tolist())
    print("── Generated text " + "─" * 50)
    print(generated_text)


if __name__ == "__main__":
    main()
