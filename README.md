# GPT From Scratch

A clean, professional implementation of a character-level **GPT language model** built from scratch with PyTorch, following Andrej Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html) series.

---

## Project Structure

```
.
├── train.py                   # Training entry point
├── generate.py                # Text generation entry point
├── requirements.txt
├── pyproject.toml
│
├── config/
│   └── config.yaml            # All hyperparameters & paths
│
├── data/
│   ├── raw/
│   │   └── input.txt          # Raw training corpus
│   └── processed/
│       └── tokenizer.json     # Saved tokenizer (auto-generated)
│
├── src/
│   ├── config.py              # Typed config dataclasses + YAML loader
│   ├── model/
│   │   ├── attention.py       # Head, MultiHeadAttention
│   │   ├── feedforward.py     # FeedForward MLP
│   │   ├── block.py           # Transformer Block (attn + FFN + residuals)
│   │   └── gpt.py             # GPTLanguageModel
│   ├── data/
│   │   └── dataset.py         # TextDataset + batch sampler
│   ├── utils/
│   │   └── tokenizer.py       # CharTokenizer (encode / decode / save / load)
│   └── training/
│       └── trainer.py         # Training loop, evaluation, checkpointing
│
├── notebooks/
│   └── gpt_dev.ipynb          # Interactive development notebook
│
├── checkpoints/               # Saved model checkpoints (git-ignored)
└── logs/                      # Training logs (git-ignored)
```

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Download training data

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
     -O data/raw/input.txt
```

### 3 – Train

```bash
python train.py
# or with a custom config:
python train.py --config config/config.yaml
```

Checkpoints are saved to `checkpoints/` after every `eval_interval` steps.

### 4 – Generate text

```bash
python generate.py --tokens 500
# with a seed prompt:
python generate.py --prompt "To be or not to be" --tokens 300
# from a specific checkpoint:
python generate.py --checkpoint checkpoints/ckpt_step04999.pt --tokens 500
```

---

## Configuration

All hyperparameters live in `config/config.yaml`:

| Section    | Key              | Default | Description                        |
|------------|------------------|---------|------------------------------------|
| `model`    | `n_embd`         | 384     | Embedding / model dimension        |
| `model`    | `n_head`         | 6       | Number of attention heads          |
| `model`    | `n_layer`        | 6       | Number of Transformer blocks       |
| `model`    | `block_size`     | 256     | Context window (max sequence len)  |
| `model`    | `dropout`        | 0.2     | Dropout probability                |
| `training` | `batch_size`     | 64      | Parallel sequences per step        |
| `training` | `max_iters`      | 5000    | Total training iterations          |
| `training` | `learning_rate`  | 3e-4    | AdamW learning rate                |
| `training` | `eval_interval`  | 500     | Steps between evaluations          |
| `data`     | `train_split`    | 0.9     | Fraction used for training         |
| —          | `device`         | `auto`  | `auto` / `cuda` / `cpu`           |

---

## Model Architecture

```
Input tokens (B, T)
    │
    ├─ Token Embedding  (vocab_size → n_embd)
    ├─ Position Embedding (block_size → n_embd)
    │
    └─ [× n_layer] Transformer Block
            ├─ LayerNorm
            ├─ Multi-Head Causal Self-Attention  (n_head heads)
            ├─ Residual add
            ├─ LayerNorm
            ├─ Feed-Forward MLP  (n_embd → 4·n_embd → n_embd)
            └─ Residual add
    │
    ├─ Final LayerNorm
    └─ Linear head  (n_embd → vocab_size)
```

---

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., *Language Models are Unsupervised Multitask Learners* – GPT-2 (2019)
- Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2023)
