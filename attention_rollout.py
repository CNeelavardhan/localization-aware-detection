"""Generate Attention Rollout localization heatmaps for the ViT branch of the Hybrid CNN-ViT
fusion classifier.

Per ATTENTION_ROLLOUT_SPEC.md: explains outputs/hybrid_fusion/best_model.pth's frozen ViT-B/16
branch via Attention Rollout (Abnar & Zuidema, 2020) - a class-agnostic, gradient-free method that
propagates the ViT's own attention weights across all 12 transformer layers (with the residual
connection each block actually has folded in), rather than backpropagating a target-class score
the way gradcam.py's Grad-CAM++ does. Runs over the full test split (352 images, real + fake), the
same set gradcam.py already processed, so vit_rollout heatmaps line up one-to-one with vit_cam for
later comparison (localization_eval.py, object_bias_floor.py).

Usage:
    python attention_rollout.py
    python attention_rollout.py --data-dir data --splits-dir data/splits --output-dir outputs/attention_rollout
    python attention_rollout.py --dry-run

Requires:
    pip install torch torchvision matplotlib pillow
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import AttentionRolloutConfig, load_attention_rollout_config, load_hybrid_fusion_config
from dataset import LABEL_NAMES, build_dataset
from gradcam import HeatmapVisualizer
from train_hybrid_fusion import HybridFusionClassifier
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("attention_rollout")


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


class AttentionCapture:
    """Temporarily patches each EncoderBlock in a torchvision ViT encoder so its forward pass
    requests attention weights (need_weights=True, average_attn_weights=True - mean-across-heads
    fusion per ATTENTION_ROLLOUT_SPEC.md) instead of discarding them (the library default,
    need_weights=False). The patched forward reproduces EncoderBlock.forward's body exactly
    (ln_1 -> self_attention -> dropout -> residual -> ln_2 -> mlp -> residual); only the
    self_attention call's need_weights/average_attn_weights arguments and the side effect of
    appending the resulting (1, 197, 197) matrix to `captured` differ. Original forward methods
    are restored on exit, so this has no effect on the model outside the `with` block, and none
    on gradcam.py's or the training code's use of the same encoder.
    """

    def __init__(self, encoder_layers: torch.nn.Sequential):
        self._layers = list(encoder_layers)
        self._originals: list = []
        self.captured: list[torch.Tensor] = []

    def _make_patched_forward(self, block: torch.nn.Module):
        def patched_forward(input: torch.Tensor) -> torch.Tensor:
            x = block.ln_1(input)
            x, weights = block.self_attention(x, x, x, need_weights=True, average_attn_weights=True)
            self.captured.append(weights.detach())
            x = block.dropout(x)
            x = x + input
            y = block.ln_2(x)
            y = block.mlp(y)
            return x + y

        return patched_forward

    def __enter__(self) -> "AttentionCapture":
        self.captured = []
        for block in self._layers:
            self._originals.append(block.forward)
            block.forward = self._make_patched_forward(block)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for block, original in zip(self._layers, self._originals):
            block.forward = original
        self._originals = []


class AttentionRollout:
    """Rollout computation (Abnar & Zuidema, 2020) over captured per-layer attention matrices."""

    @staticmethod
    def compute(layer_attentions: list[torch.Tensor]) -> torch.Tensor:
        """layer_attentions: list of 12 tensors, each (1, 197, 197), layer 0 first.
        Returns the (1, 197, 197) rolled-up matrix.
        """
        rollout = None
        for attn in layer_attentions:
            seq_len = attn.shape[-1]
            identity = torch.eye(seq_len, device=attn.device).unsqueeze(0)
            attn_hat = 0.5 * attn + 0.5 * identity
            attn_hat = attn_hat / attn_hat.sum(dim=-1, keepdim=True)
            rollout = attn_hat if rollout is None else torch.matmul(attn_hat, rollout)
        return rollout

    @staticmethod
    def cls_to_patch_grid(rollout: torch.Tensor) -> torch.Tensor:
        """rollout: (1, 197, 197). Returns the CLS row's 196 patch-position values as a
        (1, 1, 14, 14) grid (excludes column 0, the CLS-to-CLS entry).
        """
        cls_row = rollout[:, 0, 1:]  # (1, 196)
        n = cls_row.shape[0]
        grid_size = int(round(cls_row.shape[1] ** 0.5))
        return cls_row.reshape(n, 1, grid_size, grid_size)

    @staticmethod
    def normalize_and_resize(grid: torch.Tensor, size: int = 224) -> np.ndarray:
        resized = F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=False)
        arr = resized.squeeze().cpu().numpy()
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max - arr_min > 1e-8:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        return arr


class AttentionRolloutGenerator:
    def __init__(self, model: HybridFusionClassifier, device: torch.device):
        self._model = model
        self._device = device

    def generate(self, image: torch.Tensor) -> np.ndarray:
        """image: (1, 3, 224, 224), already normalized. Returns vit_rollout as a 224x224 float
        array in [0, 1]. Class-agnostic: no backward pass, no target logit.
        """
        self._model.eval()
        image = image.to(self._device)
        vit_backbone = self._model.vit_backbone

        with torch.no_grad(), AttentionCapture(vit_backbone.encoder.layers) as capture:
            vit_backbone(image)

        rollout = AttentionRollout.compute(capture.captured)
        grid = AttentionRollout.cls_to_patch_grid(rollout)
        return AttentionRollout.normalize_and_resize(grid)


class AttentionRolloutRunner:
    def __init__(self, config: AttentionRolloutConfig):
        self._config = config

    def run(self, dry_run: bool = False) -> int:
        config = self._config
        set_seed(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Using device: %s", device)

        fusion_config = load_hybrid_fusion_config({"data_dir": str(config.data_dir), "splits_dir": str(config.splits_dir)})
        model = HybridFusionClassifier(fusion_config)
        model.load_state_dict(torch.load(config.hybrid_fusion_checkpoint, map_location=device))
        model.to(device)
        model.eval()

        dataset = build_dataset(fusion_config, "test")
        LOGGER.info("Test samples: %d", len(dataset))

        generator = AttentionRolloutGenerator(model, device)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        num_images = 2 if dry_run else len(dataset)
        rows: list[dict] = []

        for idx in range(num_images):
            image_tensor, label, image_id = dataset[idx]
            ref = dataset.sample_ref(idx)
            label_name = LABEL_NAMES[label]

            vit_rollout = generator.generate(image_tensor.unsqueeze(0))
            stem = f"{image_id}_{label_name}"

            original_image = Image.open(ref.path).convert("RGB")
            overlay_path = config.output_dir / "test" / f"{stem}_rollout.png"
            HeatmapVisualizer.save_overlay(original_image, vit_rollout, overlay_path, config.overlay_alpha)

            npz_path = config.output_dir / "test" / f"{stem}_raw.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, vit_rollout=vit_rollout, y_true=label)

            rows.append(
                {
                    "image_id": image_id, "file_name": ref.file_name, "split": "test",
                    "y_true": label, "rollout_path": str(overlay_path), "raw_npz_path": str(npz_path),
                }
            )

            if (idx + 1) % 25 == 0 or (idx + 1) == num_images:
                LOGGER.info("Processed %d/%d images", idx + 1, num_images)

        self._write_summary(rows)
        if not dry_run:
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)
        LOGGER.info("Attention rollout generation complete. Outputs written to %s", config.output_dir)
        return 0

    def _write_summary(self, rows: list[dict]) -> None:
        if not rows:
            return
        summary_path = self._config.output_dir / self._config.summary_filename
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote summary for %d images to %s", len(rows), summary_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--hybrid-fusion-checkpoint", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_attention_rollout_config(
        {
            "data_dir": args.data_dir,
            "splits_dir": args.splits_dir,
            "output_dir": args.output_dir,
            "hybrid_fusion_checkpoint": args.hybrid_fusion_checkpoint,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / "attention_rollout.log")
    LOGGER.info("Loaded configuration (seed=%d)", config.seed)

    if not config.hybrid_fusion_checkpoint.exists():
        LOGGER.error("Hybrid fusion checkpoint not found: %s", config.hybrid_fusion_checkpoint)
        return 1

    try:
        return AttentionRolloutRunner(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during attention rollout generation:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
