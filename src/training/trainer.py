"""
Trainer.

Encapsulates the full training loop, periodic evaluation, logging, and
checkpoint management.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from src.config import Config
from src.data.dataset import TextDataset


class Trainer:
    """
    Manages training a GPTLanguageModel.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    dataset : TextDataset
    config : Config
    device : str
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: TextDataset,
        config: Config,
        device: str,
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.dataset   = dataset
        self.config    = config
        self.device    = device

        self._history: list[dict] = []   # training log

    # ------------------------------------------------------------------
    # Loss estimation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """
        Evaluate mean loss on `eval_iters` batches for each split.

        Returns a dict with keys ``"train"`` and ``"val"``.
        """
        results = {}
        self.model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(self.config.training.eval_iters)
            for k in range(self.config.training.eval_iters):
                X, Y = self.dataset.get_batch(
                    split,
                    self.config.training.batch_size,
                    self.config.model.block_size,
                    self.device,
                )
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            results[split] = losses.mean().item()
        self.model.train()
        return results

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self) -> list[dict]:
        """
        Run the full training loop.

        Returns
        -------
        list[dict] – history of evaluation metrics recorded during training.
        """
        cfg_t = self.config.training
        cfg_m = self.config.model

        print(f"Training for {cfg_t.max_iters:,} iterations …")
        t0 = time.time()

        for step in range(cfg_t.max_iters):
            # periodic evaluation 
            if step % cfg_t.eval_interval == 0 or step == cfg_t.max_iters - 1:
                losses = self.estimate_loss()
                elapsed = time.time() - t0
                print(
                    f"step {step:>5d}/{cfg_t.max_iters} | "
                    f"train loss {losses['train']:.4f} | "
                    f"val loss {losses['val']:.4f} | "
                    f"elapsed {elapsed:.1f}s"
                )
                self._history.append({"step": step, **losses})

                #  auto-save checkpoint 
                ckpt_path = (
                    Path(self.config.paths.checkpoints_dir)
                    / f"ckpt_step{step:05d}.pt"
                )
                self.save_checkpoint(str(ckpt_path), step=step, loss=losses["val"])

            #  training step 
            xb, yb = self.dataset.get_batch(
                "train",
                cfg_t.batch_size,
                cfg_m.block_size,
                self.device,
            )
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        print(f"Training complete in {time.time() - t0:.1f}s")
        return self._history

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        path: str,
        step: int = 0,
        loss: float = float("inf"),
    ) -> None:
        """Save model + optimizer state to `path`."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "val_loss": loss,
                "model_config": self.config.model,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        """
        Load model + optimizer state from `path`.

        Returns
        -------
        int – the step at which the checkpoint was saved.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt.get("step", 0)
        loss = ckpt.get("val_loss", float("inf"))
        print(f"Loaded checkpoint from '{path}' (step={step}, val_loss={loss:.4f})")
        return step
