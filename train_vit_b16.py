"""Train the ViT-B/16 binary (real vs. manipulated) classifier.

Per VIT_B16_SPEC.md: ImageNet-pretrained ViT-B/16 backbone, CLS token pooling (ViT's
equivalent of EfficientNet-B3's global average pooling), dropout, dense head, single sigmoid
output. AdamW with discriminative learning rates, cosine-annealed with warmup, early stopping
on validation F1. At the end, evaluates the best checkpoint on train/val/test and writes
predictions, metrics, and feature vectors (for the later CNN-ViT fusion phase).

Usage:
    python train_vit_b16.py
    python train_vit_b16.py --data-dir data --output-dir outputs/vit_b16
    python train_vit_b16.py --dry-run

Requires:
    pip install torch torchvision scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_16_Weights, vit_b_16

from config import ViTB16Config, load_vit_b16_config
from dataset import build_dataset
from evaluate import ModelEvaluator, ResultWriter
from training_utils import (
    CosineWarmupScheduler,
    EarlyStopping,
    EpochHistory,
    ReproducibilityRecorder,
    set_seed,
)

LOGGER = logging.getLogger("train_vit_b16")


def configure_logging(log_level: str, log_file: Path) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


class ViTB16Classifier(nn.Module):
    """Torchvision ViT-B/16: CLS token (768-dim) -> Dropout -> Dense -> ReLU -> Dropout -> 1 logit."""

    def __init__(self, dropout: float, dense_units: int, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vit_b_16(weights=weights)
        self.conv_proj = backbone.conv_proj
        self.class_token = backbone.class_token
        self.encoder = backbone.encoder
        in_features = backbone.hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(dense_units, 1),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)  # (n, hidden_dim, n_h, n_w)
        n, hidden_dim, n_h, n_w = x.shape
        x = x.reshape(n, hidden_dim, n_h * n_w).permute(0, 2, 1)  # (n, n_h*n_w, hidden_dim)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self._patchify(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        feats = x[:, 0]
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits

    def parameter_groups(self, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        backbone_params = list(self.conv_proj.parameters()) + [self.class_token] + list(self.encoder.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": self.classifier.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ]


class Trainer:
    def __init__(self, config: ViTB16Config):
        self._config = config

    def run(self, dry_run: bool = False) -> int:
        config = self._config
        set_seed(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Using device: %s", device)

        train_dataset = build_dataset(config, "train")
        val_dataset = build_dataset(config, "val")
        LOGGER.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        )

        model = ViTB16Classifier(dropout=config.dropout, dense_units=config.dense_units)
        model.to(device)

        optimizer = AdamW(model.parameter_groups(config.backbone_lr, config.head_lr, config.weight_decay))
        scheduler = CosineWarmupScheduler(optimizer, config.warmup_epochs, config.max_epochs)
        criterion = BCEWithLogitsLoss()
        early_stopping = EarlyStopping(config.early_stopping_patience)
        history = EpochHistory()

        config.output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = config.output_dir / config.best_model_filename

        if dry_run:
            return self._dry_run(model, train_loader, criterion, optimizer, device)

        for epoch in range(1, config.max_epochs + 1):
            train_loss, train_acc = self._train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_result = ModelEvaluator(model, device).evaluate(val_loader, criterion=criterion)
            val_loss = val_result.loss

            current_lr = optimizer.param_groups[0]["lr"]
            LOGGER.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f "
                "val_f1=%.4f val_precision=%.4f val_recall=%.4f val_roc_auc=%.4f | lr=%.2e",
                epoch, config.max_epochs, train_loss, train_acc, val_loss, val_result.metrics.accuracy,
                val_result.metrics.f1, val_result.metrics.precision, val_result.metrics.recall,
                val_result.metrics.roc_auc, current_lr,
            )

            history.add(
                epoch=epoch, train_loss=train_loss, train_accuracy=train_acc, val_loss=val_loss,
                val_accuracy=val_result.metrics.accuracy, val_f1=val_result.metrics.f1,
                val_precision=val_result.metrics.precision, val_recall=val_result.metrics.recall,
                val_roc_auc=val_result.metrics.roc_auc, lr=current_lr,
            )

            monitored_score = getattr(val_result.metrics, config.early_stopping_metric)
            if early_stopping.step(monitored_score):
                torch.save(model.state_dict(), best_model_path)
                LOGGER.info("New best val %s=%.4f — saved checkpoint to %s", config.early_stopping_metric, monitored_score, best_model_path)

            scheduler.step()

            if early_stopping.should_stop:
                LOGGER.info("Early stopping triggered after %d epoch(s) without improvement", config.early_stopping_patience)
                break

        history.write_csv(config.output_dir / config.epoch_history_filename)
        history.plot_curves(config.output_dir / config.curves_filename)
        ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)

        if not best_model_path.exists():
            LOGGER.error("No checkpoint was ever saved (training may have produced 0 epochs)")
            return 1

        self._final_evaluation(config, best_model_path, device)
        LOGGER.info("Training complete. Outputs written to %s", config.output_dir)
        return 0

    def _train_one_epoch(self, model, loader, criterion, optimizer, device) -> tuple[float, float]:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / max(1, total), correct / max(1, total)

    def _dry_run(self, model, loader, criterion, optimizer, device) -> int:
        model.train()
        images, labels, ids = next(iter(loader))
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        logits, feats = model(images, return_features=True)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        LOGGER.info(
            "[DRY RUN] batch_images=%s logits=%s features=%s loss=%.4f",
            tuple(images.shape), tuple(logits.shape), tuple(feats.shape), loss.item(),
        )
        return 0

    def _final_evaluation(self, config: ViTB16Config, checkpoint_path: Path, device) -> None:
        model = ViTB16Classifier(dropout=config.dropout, dense_units=config.dense_units)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)

        for split in ("train", "val", "test"):
            dataset = build_dataset(config, split)
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
            result = ModelEvaluator(model, device).evaluate(loader)

            LOGGER.info(
                "[FINAL] split=%s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
                split, result.metrics.accuracy, result.metrics.precision,
                result.metrics.recall, result.metrics.f1, result.metrics.roc_auc,
            )
            LOGGER.info("[FINAL] split=%s confusion_matrix=%s", split, result.metrics.confusion_matrix)

            ResultWriter.write_predictions(config.output_dir / f"{split}_{config.predictions_filename}", result, split)
            ResultWriter.write_metrics(config.output_dir / f"{split}_{config.metrics_filename}", result.metrics)
            ResultWriter.write_features(config.output_dir / f"{split}_{config.features_filename}", result)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_vit_b16_config(
        {
            "data_dir": args.data_dir,
            "splits_dir": args.splits_dir,
            "output_dir": args.output_dir,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "num_workers": args.num_workers,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / config.training_log_filename)
    LOGGER.info("Loaded configuration (seed=%d, batch_size=%d, max_epochs=%d)", config.seed, config.batch_size, config.max_epochs)

    try:
        return Trainer(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during training:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
