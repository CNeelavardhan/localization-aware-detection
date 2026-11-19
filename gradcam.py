"""Generate Grad-CAM++ localization heatmaps for the Hybrid CNN-ViT fusion classifier.

Per GRADCAM_SPEC.md: explains outputs/hybrid_fusion/best_model.pth's "manipulated" prediction by
backpropagating the fused logit through both frozen backbones, producing a CNN-branch heatmap
(EfficientNet-B3's last conv block, 7x7), a ViT-branch heatmap (ViT-B/16's last-layer patch
tokens, 14x14), and a combined hybrid heatmap (elementwise average of both, upsampled to 224x224).
Runs over the full test split (352 images, real + fake) so Phase 7 (Localization Evaluation) can
score the fake-image heatmaps against data/masks/ ground truth without re-running Grad-CAM++.

Usage:
    python gradcam.py
    python gradcam.py --data-dir data --splits-dir data/splits --output-dir outputs/gradcam
    python gradcam.py --dry-run

Requires:
    pip install torch torchvision matplotlib pillow
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import GradCAMConfig, load_gradcam_config, load_hybrid_fusion_config
from dataset import LABEL_NAMES, build_dataset
from train_hybrid_fusion import HybridFusionClassifier
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("gradcam")


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


class GradCAMPlusPlus:
    """Grad-CAM++ (Chattopadhay et al., 2018), implemented directly against
    HybridFusionClassifier.forward_with_activations() — one branch-agnostic weighted-activation
    computation reused for both the CNN's (N,1536,7,7) and the ViT's (N,768,14,14) target layers.
    """

    def __init__(self, model: HybridFusionClassifier, device: torch.device):
        self._model = model
        self._device = device

    def generate(self, image: torch.Tensor) -> dict:
        """image: (1, 3, 224, 224), already normalized. Returns cnn_cam/vit_cam/hybrid_cam as
        224x224 float arrays in [0, 1], plus the model's y_prob for this image.
        """
        self._model.eval()
        image = image.to(self._device)
        image.requires_grad_(True)  # forces the frozen-backbone subgraph to build gradients

        self._model.zero_grad(set_to_none=True)
        logits, cnn_spatial, vit_layer_input = self._model.forward_with_activations(image)
        score = logits.squeeze()
        score.backward()

        cnn_cam = self._compute_camplusplus(cnn_spatial.detach(), cnn_spatial.grad.detach())

        vit_activations, vit_grads = self._patch_tokens_to_grid(vit_layer_input.detach(), vit_layer_input.grad.detach())
        vit_cam = self._compute_camplusplus(vit_activations, vit_grads)

        cnn_cam_norm = self._normalize_and_resize(cnn_cam)
        vit_cam_norm = self._normalize_and_resize(vit_cam)
        hybrid_cam = (cnn_cam_norm + vit_cam_norm) / 2.0

        y_prob = torch.sigmoid(logits).item()
        return {"cnn_cam": cnn_cam_norm, "vit_cam": vit_cam_norm, "hybrid_cam": hybrid_cam, "y_prob": y_prob}

    @staticmethod
    def _patch_tokens_to_grid(activations: torch.Tensor, gradients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, 197, 768) -> drop the CLS position (index 0), reshape the 196 patch tokens to a
        14x14 grid: (n, 768, 14, 14), matching the CNN branch's (n, C, H, W) convention.
        """
        patch_activations = activations[:, 1:]
        patch_gradients = gradients[:, 1:]
        n, num_patches, hidden_dim = patch_activations.shape
        grid = int(round(num_patches ** 0.5))
        spatial_activations = patch_activations.permute(0, 2, 1).reshape(n, hidden_dim, grid, grid)
        spatial_gradients = patch_gradients.permute(0, 2, 1).reshape(n, hidden_dim, grid, grid)
        return spatial_activations, spatial_gradients

    @staticmethod
    def _compute_camplusplus(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """activations, gradients: (1, C, H, W). Returns cam: (1, H, W), un-normalized, ReLU'd."""
        eps = 1e-7
        grads_power_2 = gradients ** 2
        grads_power_3 = grads_power_2 * gradients
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)

        alpha_denom = grads_power_2 * 2.0 + sum_activations * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.full_like(alpha_denom, eps))
        alphas = grads_power_2 / alpha_denom

        weights = (alphas * torch.relu(gradients)).sum(dim=(2, 3))  # (1, C)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1)  # (1, H, W)
        return torch.relu(cam)

    @staticmethod
    def _normalize_and_resize(cam: torch.Tensor, size: int = 224) -> np.ndarray:
        cam = F.interpolate(cam.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


class HeatmapVisualizer:
    @staticmethod
    def save_overlay(original_image: Image.Image, cam: np.ndarray, path: Path, alpha: float) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        base = np.array(original_image.resize((224, 224))).astype(np.float64) / 255.0
        heatmap = cm.jet(cam)[:, :, :3]
        overlay = np.clip((1 - alpha) * base + alpha * heatmap, 0.0, 1.0)

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, overlay)


class GradCAMRunner:
    def __init__(self, config: GradCAMConfig):
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

        cam_generator = GradCAMPlusPlus(model, device)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        num_images = 2 if dry_run else len(dataset)
        rows: list[dict] = []

        for idx in range(num_images):
            image_tensor, label, image_id = dataset[idx]
            ref = dataset.sample_ref(idx)
            label_name = LABEL_NAMES[label]

            result = cam_generator.generate(image_tensor.unsqueeze(0))
            stem = f"{image_id}_{label_name}"

            original_image = Image.open(ref.path).convert("RGB")
            paths = {}
            for cam_name in ("cnn_cam", "vit_cam", "hybrid_cam"):
                overlay_path = config.output_dir / "test" / f"{stem}_{cam_name}.png"
                HeatmapVisualizer.save_overlay(original_image, result[cam_name], overlay_path, config.overlay_alpha)
                paths[cam_name] = overlay_path

            npz_path = config.output_dir / "test" / f"{stem}_raw.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                cnn_cam=result["cnn_cam"], vit_cam=result["vit_cam"], hybrid_cam=result["hybrid_cam"],
                y_true=label, y_prob=result["y_prob"],
            )

            rows.append(
                {
                    "image_id": image_id, "file_name": ref.file_name, "split": "test",
                    "y_true": label, "y_pred": int(result["y_prob"] >= 0.5), "y_prob": f"{result['y_prob']:.6f}",
                    "cnn_cam_path": str(paths["cnn_cam"]), "vit_cam_path": str(paths["vit_cam"]),
                    "hybrid_cam_path": str(paths["hybrid_cam"]), "raw_npz_path": str(npz_path),
                }
            )

            if (idx + 1) % 25 == 0 or (idx + 1) == num_images:
                LOGGER.info("Processed %d/%d images", idx + 1, num_images)

        self._write_summary(rows)
        if not dry_run:
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)
        LOGGER.info("Grad-CAM++ generation complete. Outputs written to %s", config.output_dir)
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
    config = load_gradcam_config(
        {
            "data_dir": args.data_dir,
            "splits_dir": args.splits_dir,
            "output_dir": args.output_dir,
            "hybrid_fusion_checkpoint": args.hybrid_fusion_checkpoint,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / "gradcam.log")
    LOGGER.info("Loaded configuration (seed=%d)", config.seed)

    if not config.hybrid_fusion_checkpoint.exists():
        LOGGER.error("Hybrid fusion checkpoint not found: %s", config.hybrid_fusion_checkpoint)
        return 1

    try:
        return GradCAMRunner(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during Grad-CAM++ generation:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
