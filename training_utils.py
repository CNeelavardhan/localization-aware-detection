"""Model-agnostic training infrastructure shared across the binary-classifier training phases
(EfficientNet-B3, ViT-B/16, and any future backbone trained the same way).
"""

from __future__ import annotations

import csv
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CosineWarmupScheduler:
    """Linear warmup for `warmup_epochs`, then cosine decay to 0 over the remaining epochs."""

    def __init__(self, optimizer: Any, warmup_epochs: int, max_epochs: int):
        import math

        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self._scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self) -> None:
        self._scheduler.step()


class EarlyStopping:
    def __init__(self, patience: int):
        self._patience = patience
        self._best_score: Optional[float] = None
        self._num_bad_epochs = 0

    def step(self, score: float) -> bool:
        """Returns True if `score` is a new best."""
        if self._best_score is None or score > self._best_score:
            self._best_score = score
            self._num_bad_epochs = 0
            return True
        self._num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self._num_bad_epochs >= self._patience


class EpochHistory:
    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add(self, **fields: Any) -> None:
        self._rows.append(fields)

    def write_csv(self, path: Path) -> None:
        if not self._rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)

    def plot_curves(self, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in self._rows]
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

        ax_loss.plot(epochs, [r["train_loss"] for r in self._rows], label="train")
        ax_loss.plot(epochs, [r["val_loss"] for r in self._rows], label="val")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()

        ax_acc.plot(epochs, [r["train_accuracy"] for r in self._rows], label="train")
        ax_acc.plot(epochs, [r["val_accuracy"] for r in self._rows], label="val")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Accuracy")
        ax_acc.legend()

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


class ReproducibilityRecorder:
    @staticmethod
    def record(config: Any, path: Path) -> None:
        import torch
        import torchvision

        info: dict[str, Any] = {
            "seed": config.seed,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in config.__dict__.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2, default=str))
