"""Train the Hybrid CNN-ViT fusion binary (real vs. manipulated) classifier.

Per HYBRID_FUSION_SPEC.md: loads the already-trained EfficientNet-B3 and ViT-B/16 backbones
(Phase 3/4 checkpoints), freezes both, projects each backbone's pre-classifier feature vector to
a shared dimension, concatenates and fuses them through a small MLP, and trains only that fusion
head on top of the frozen backbones. At the end, evaluates the best checkpoint on train/val/test
and writes predictions, metrics, feature vectors, and a comparison against the two standalone
backbones' test metrics.

Usage:
    python train_hybrid_fusion.py
    python train_hybrid_fusion.py --data-dir data --output-dir outputs/hybrid_fusion
    python train_hybrid_fusion.py --dry-run

Requires:
    pip install torch torchvision scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import json
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
from torchvision.models import efficientnet_b3, vit_b_16

from config import HybridFusionConfig, load_hybrid_fusion_config
from dataset import build_dataset
from evaluate import ModelEvaluator, ResultWriter
from training_utils import (
    CosineWarmupScheduler,
    EarlyStopping,
    EpochHistory,
    ReproducibilityRecorder,
    set_seed,
)

LOGGER = logging.getLogger("train_hybrid_fusion")


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


class _EfficientNetB3Backbone(nn.Module):
    """Frozen EfficientNet-B3 feature extractor: features -> avgpool -> 1536-dim vector."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        backbone = efficientnet_b3(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self._load_checkpoint(checkpoint_path)
        for param in self.parameters():
            param.requires_grad = False

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        features_sd = {k[len("features."):]: v for k, v in state_dict.items() if k.startswith("features.")}
        self.features.load_state_dict(features_sd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class _ViTB16Backbone(nn.Module):
    """Frozen ViT-B/16 feature extractor: patchify -> encoder -> CLS token, 768-dim vector."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        backbone = vit_b_16(weights=None)
        self.conv_proj = backbone.conv_proj
        self.class_token = backbone.class_token
        self.encoder = backbone.encoder
        self._load_checkpoint(checkpoint_path)
        for param in self.parameters():
            param.requires_grad = False

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        conv_proj_sd = {k[len("conv_proj."):]: v for k, v in state_dict.items() if k.startswith("conv_proj.")}
        encoder_sd = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        self.conv_proj.load_state_dict(conv_proj_sd)
        self.encoder.load_state_dict(encoder_sd)
        with torch.no_grad():
            self.class_token.copy_(state_dict["class_token"])

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)
        n, hidden_dim, n_h, n_w = x.shape
        x = x.reshape(n, hidden_dim, n_h * n_w).permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._patchify(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]


class HybridFusionClassifier(nn.Module):
    """Frozen EfficientNet-B3 + frozen ViT-B/16 -> per-backbone projection -> concat -> fusion
    MLP -> single sigmoid logit. Only the projections, fusion MLP, and head are trainable.
    """

    def __init__(self, config: HybridFusionConfig):
        super().__init__()
        self.cnn_backbone = _EfficientNetB3Backbone(config.efficientnet_b3_checkpoint)
        self.vit_backbone = _ViTB16Backbone(config.vit_b16_checkpoint)

        self.cnn_proj = nn.Sequential(
            nn.Linear(config.cnn_feature_dim, config.projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(config.vit_feature_dim, config.projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.projection_dim * 2, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, 1),
        )

    def train(self, mode: bool = True) -> "HybridFusionClassifier":
        super().train(mode)
        self.cnn_backbone.eval()
        self.vit_backbone.eval()
        return self

    def forward(self, x: torch.Tensor, return_features: bool = False):
        with torch.no_grad():
            cnn_feats = self.cnn_backbone(x)
            vit_feats = self.vit_backbone(x)
        fused_in = torch.cat([self.cnn_proj(cnn_feats), self.vit_proj(vit_feats)], dim=1)
        fused = self.fusion(fused_in)
        logits = self.head(fused)
        if return_features:
            return logits, fused
        return logits

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Grad-CAM++ path (GRADCAM_SPEC.md): same computation as forward(), but without the
        no_grad() wrapper around the backbones, so gradients can flow back to each branch's
        target-layer activations. Backbone weights stay frozen; this only restores gradient flow
        through activations.

        CNN target layer: EfficientNet-B3's last conv block output (n, 1536, 7, 7) - a direct
        ancestor of the fused logit via avgpool -> cnn_proj.

        ViT target layer: the INPUT to ViT-B/16's last transformer block (n, 197, 768; CLS +
        196 patches), not the encoder's final output. The classification head only reads the
        final CLS token (index 0 of the encoder's output), so the *final* layer's patch-token
        outputs are never used downstream and their gradient would be exactly zero. The last
        block's self-attention mixes the CLS position with every patch position of its INPUT to
        produce the CLS position of its OUTPUT, so this earlier tensor is where patch-level
        gradients are actually nonzero - standard practice for gradient-based explainability on
        CLS-pooled transformers.

        Returns (logits, cnn_spatial, vit_pre_last_block) with retain_grad() already called on
        both target-layer tensors; the caller backprops from a scalar score, then reads `.grad`.
        """
        cnn_spatial = self.cnn_backbone.features(x)
        cnn_spatial.retain_grad()
        cnn_pooled = self.cnn_backbone.avgpool(cnn_spatial)
        cnn_feats = torch.flatten(cnn_pooled, 1)

        vit_patches = self.vit_backbone._patchify(x)
        n = vit_patches.shape[0]
        batch_class_token = self.vit_backbone.class_token.expand(n, -1, -1)
        vit_tokens = torch.cat([batch_class_token, vit_patches], dim=1)

        encoder = self.vit_backbone.encoder
        pos_embedded = encoder.dropout(vit_tokens + encoder.pos_embedding)
        vit_pre_last_block = encoder.layers[:-1](pos_embedded)
        vit_pre_last_block.retain_grad()
        vit_encoded = encoder.ln(encoder.layers[-1](vit_pre_last_block))
        vit_feats = vit_encoded[:, 0]

        fused_in = torch.cat([self.cnn_proj(cnn_feats), self.vit_proj(vit_feats)], dim=1)
        fused = self.fusion(fused_in)
        logits = self.head(fused)
        return logits, cnn_spatial, vit_pre_last_block

    def parameter_groups(self, learning_rate: float, weight_decay: float) -> list[dict]:
        trainable_params = (
            list(self.cnn_proj.parameters())
            + list(self.vit_proj.parameters())
            + list(self.fusion.parameters())
            + list(self.head.parameters())
        )
        return [{"params": trainable_params, "lr": learning_rate, "weight_decay": weight_decay}]


class Trainer:
    def __init__(self, config: HybridFusionConfig):
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

        model = HybridFusionClassifier(config)
        model.to(device)

        optimizer = AdamW(model.parameter_groups(config.learning_rate, config.weight_decay))
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

    def _final_evaluation(self, config: HybridFusionConfig, checkpoint_path: Path, device) -> None:
        model = HybridFusionClassifier(config)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)

        test_metrics_dict: dict[str, float] = {}
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

            if split == "test":
                test_metrics_dict = result.metrics.to_dict()

        self._write_comparison(config, test_metrics_dict)

    def _write_comparison(self, config: HybridFusionConfig, fusion_test_metrics: dict) -> None:
        comparison: dict[str, dict] = {"hybrid_fusion": fusion_test_metrics}
        for name, path in (
            ("efficientnet_b3", config.efficientnet_b3_checkpoint.parent / "test_metrics.json"),
            ("vit_b16", config.vit_b16_checkpoint.parent / "test_metrics.json"),
        ):
            if path.exists():
                comparison[name] = json.loads(path.read_text())
            else:
                LOGGER.warning("Standalone test metrics not found for comparison: %s", path)

        comparison_path = config.output_dir / config.comparison_filename
        comparison_path.write_text(json.dumps(comparison, indent=2))
        LOGGER.info("Wrote standalone-vs-fusion comparison to %s", comparison_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--efficientnet-b3-checkpoint", type=str, default=None)
    parser.add_argument("--vit-b16-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_hybrid_fusion_config(
        {
            "data_dir": args.data_dir,
            "splits_dir": args.splits_dir,
            "output_dir": args.output_dir,
            "efficientnet_b3_checkpoint": args.efficientnet_b3_checkpoint,
            "vit_b16_checkpoint": args.vit_b16_checkpoint,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "num_workers": args.num_workers,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / config.training_log_filename)
    LOGGER.info("Loaded configuration (seed=%d, batch_size=%d, max_epochs=%d)", config.seed, config.batch_size, config.max_epochs)

    for checkpoint_path in (config.efficientnet_b3_checkpoint, config.vit_b16_checkpoint):
        if not checkpoint_path.exists():
            LOGGER.error("Required backbone checkpoint not found: %s", checkpoint_path)
            return 1

    try:
        return Trainer(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during training:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
