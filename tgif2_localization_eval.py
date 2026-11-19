"""Zero-shot localization evaluation of the Hybrid CNN-ViT model on TGIF2.

Per TGIF2_LOCALIZATION_EVAL_SPEC.md: for each manipulated image in manifests/tgif2_manifest.csv
(var_id=0 only, both testing+validation splits - 4,788 images), generates all four established
heatmap types (cnn_cam/vit_cam/hybrid_cam via Grad-CAM++, vit_rollout via Attention Rollout) and
scores each against its ground-truth mask, resized to 224x224 exactly as localization_eval.py
already does for v1. No training happens against TGIF2 anywhere in this script - the frozen
outputs/hybrid_fusion/best_model.pth checkpoint is used exactly as-is.

gradcam.py, attention_rollout.py, and localization_eval.py are all FROZEN and are only imported
from here, never modified, per this project's spec convention.

Usage:
    python tgif2_localization_eval.py
    python tgif2_localization_eval.py --dry-run

Requires:
    pip install torch torchvision scikit-learn numpy pillow
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from attention_rollout import AttentionRolloutGenerator
from config import Tgif2LocalizationEvalConfig, load_hybrid_fusion_config, load_tgif2_localization_eval_config
from dataset import TransformFactory
from gradcam import GradCAMPlusPlus, HeatmapVisualizer
from localization_eval import HEATMAP_TYPES as GRADCAM_HEATMAP_TYPES  # cnn_cam, vit_cam, hybrid_cam
from localization_eval import LocalizationMetricsCalculator, MaskLoader
from PIL import Image
from train_hybrid_fusion import HybridFusionClassifier
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("tgif2_localization_eval")

ALL_HEATMAP_TYPES = (*GRADCAM_HEATMAP_TYPES, "vit_rollout")

ORIG_RE = re.compile(r"^(\d+)_orig\.png$")
MASK_RE = re.compile(r"^(\d+)_mask(?:_(bbox|segm|random))?\.png$")

SHARE_ORIG_DIRS = {
    "tgif": Path("tgif-images/orig"),
    "flux": Path("tgif-flux/orig-flux"),
    "random": Path("tgif-images/orig"),  # random share reuses the tgif share's authentic images
}
SHARE_MASK_DIRS = {
    "tgif": Path("tgif-images/masks"),
    "flux": Path("tgif-flux/masks-flux"),
}
RANDOM_MASK_DIRS = {
    "sdxl": Path("tgif-random64_60p/masks-sdxl"),
    "sd2": Path("tgif-random64_60p/masks-sd2"),
    "flux": Path("tgif-random64_60p/masks-flux"),
}


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


class Tgif2ManifestLoader:
    """Reads manifests/tgif2_manifest.csv and resolves each row's authentic-image and mask paths,
    via one-time indexes built by walking each orig/mask directory (not per-row globbing - the
    manifest has thousands of rows).
    """

    def __init__(self, tgif_dataset_dir: Path):
        self._root = tgif_dataset_dir
        self._orig_index = {
            share: self._build_orig_index(self._root / rel_dir) for share, rel_dir in SHARE_ORIG_DIRS.items()
        }
        self._mask_index = {
            "tgif": self._build_mask_index(self._root / SHARE_MASK_DIRS["tgif"]),
            "flux": self._build_mask_index(self._root / SHARE_MASK_DIRS["flux"]),
        }
        self._random_mask_index = {
            gen_model: self._build_mask_index(self._root / rel_dir) for gen_model, rel_dir in RANDOM_MASK_DIRS.items()
        }

    @staticmethod
    def _build_orig_index(orig_dir: Path) -> dict:
        index = {}
        for split in ("testing", "validation"):
            for path in (orig_dir / split).rglob("*_orig.png"):
                m = ORIG_RE.match(path.name)
                if m:
                    index[(split, m.group(1))] = path
        return index

    @staticmethod
    def _build_mask_index(mask_dir: Path) -> dict:
        index = {}
        for split in ("testing", "validation"):
            d = mask_dir / split
            if not d.exists():
                continue
            for path in d.rglob("*.png"):
                m = MASK_RE.match(path.name)
                if m:
                    index[(split, m.group(1), m.group(2) or "full")] = path
        return index

    def resolve(self, row: dict) -> Optional[tuple[Path, Path]]:
        share, split, coco_id, mask_type, gen_model = (
            row["source_share"], row["split"], row["coco_id"], row["mask_type"], row["gen_model"],
        )
        orig_path = self._orig_index.get(share, {}).get((split, coco_id))
        if share == "random":
            mask_path = self._random_mask_index.get(gen_model, {}).get((split, coco_id, mask_type))
        else:
            mask_path = self._mask_index.get(share, {}).get((split, coco_id, mask_type))
        if orig_path is None or mask_path is None:
            return None
        return orig_path, mask_path

    def load_rows(self, manifest_path: Path, var_id_filter: str) -> list[dict]:
        with manifest_path.open("r", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["var_id"] == var_id_filter]
        return rows


class Tgif2LocalizationRunner:
    def __init__(self, config: Tgif2LocalizationEvalConfig):
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

        transform = TransformFactory.build(fusion_config, train=False)
        cam_generator = GradCAMPlusPlus(model, device)
        rollout_generator = AttentionRolloutGenerator(model, device)

        manifest_loader = Tgif2ManifestLoader(config.tgif_dataset_dir)
        rows = manifest_loader.load_rows(config.manifest_path, config.var_id_filter)
        LOGGER.info("Loaded %d manifest rows at var_id=%s", len(rows), config.var_id_filter)

        if dry_run:
            rows = rows[:3]

        config.output_dir.mkdir(parents=True, exist_ok=True)
        npz_dir = config.output_dir / "tgif2_gradcam"

        per_image_rows: list[dict] = []
        unresolved = 0

        for idx, row in enumerate(rows):
            resolved = manifest_loader.resolve(row)
            if resolved is None:
                unresolved += 1
                LOGGER.warning("Could not resolve orig/mask for row: %s", row)
                continue
            _orig_path, mask_path = resolved
            manipulated_path = Path(row["file_path"])

            with Image.open(manipulated_path) as img:
                image_tensor = transform(img.convert("RGB")).unsqueeze(0)
            mask = MaskLoader.load_binary_mask(mask_path)

            cam_result = cam_generator.generate(image_tensor)
            vit_rollout = rollout_generator.generate(image_tensor)
            heatmaps = {
                "cnn_cam": cam_result["cnn_cam"], "vit_cam": cam_result["vit_cam"],
                "hybrid_cam": cam_result["hybrid_cam"], "vit_rollout": vit_rollout,
            }

            stem = f"{row['coco_id']}_{row['mode']}_{row['gen_model']}_{row['mask_type']}"
            npz_path = npz_dir / row["source_share"] / row["split"] / f"{stem}_raw.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, **heatmaps, y_prob=cam_result["y_prob"])

            result_row = {
                "coco_id": row["coco_id"], "mask_type": row["mask_type"], "gen_model": row["gen_model"],
                "var_id": row["var_id"], "mode": row["mode"], "source_share": row["source_share"],
                "split": row["split"], "y_prob": f"{cam_result['y_prob']:.6f}", "raw_npz_path": str(npz_path),
            }
            for heatmap_type in ALL_HEATMAP_TYPES:
                metrics = LocalizationMetricsCalculator.compute(mask, heatmaps[heatmap_type], config.binarization_threshold)
                for field, value in (metrics.to_dict().items() if metrics is not None else []):
                    result_row[f"{heatmap_type}_{field}"] = value
            per_image_rows.append(result_row)

            if (idx + 1) % 200 == 0 or (idx + 1) == len(rows):
                LOGGER.info("Processed %d/%d images (%d unresolved)", idx + 1, len(rows), unresolved)

        self._write_per_image_csv(per_image_rows)
        self._write_metrics_json(per_image_rows, len(rows), unresolved)

        if not dry_run:
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)

        LOGGER.info("TGIF2 localization evaluation complete. Outputs written to %s", config.output_dir)
        return 0

    def _write_per_image_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        path = self._config.output_dir / self._config.per_image_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote per-image localization metrics for %d images to %s", len(rows), path)

    def _write_metrics_json(self, rows: list[dict], num_candidates: int, unresolved: int) -> None:
        def _aggregate(subset: list[dict]) -> dict:
            agg = {}
            for heatmap_type in ALL_HEATMAP_TYPES:
                key = f"{heatmap_type}_iou"
                values = [float(r[key]) for r in subset if r.get(key) not in (None, "")]
                if not values:
                    agg[heatmap_type] = {"num_scored": 0}
                    continue
                agg[heatmap_type] = {
                    "num_scored": len(values),
                    "mean_iou": float(np.mean(values)),
                    "median_iou": float(np.median(values)),
                    "mean_auc": float(np.mean([float(r[f"{heatmap_type}_auc"]) for r in subset if r.get(f"{heatmap_type}_auc") not in (None, "")])),
                }
            return agg

        result = {
            "num_candidate_images": num_candidates, "num_unresolved": unresolved, "num_scored": len(rows),
            "pooled": _aggregate(rows),
            "by_mask_type": {mt: _aggregate([r for r in rows if r["mask_type"] == mt]) for mt in sorted({r["mask_type"] for r in rows})},
            "by_share_mode_genmodel": {
                f"{share}/{mode}/{gen}": _aggregate([r for r in rows if (r["source_share"], r["mode"], r["gen_model"]) == (share, mode, gen)])
                for share, mode, gen in sorted({(r["source_share"], r["mode"], r["gen_model"]) for r in rows})
            },
        }
        path = self._config.output_dir / self._config.metrics_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        LOGGER.info("Wrote aggregate metrics to %s", path)
        for heatmap_type, agg in result["pooled"].items():
            if agg.get("num_scored"):
                LOGGER.info("[pooled/%s] n=%d mean_iou=%.4f median_iou=%.4f mean_auc=%.4f",
                            heatmap_type, agg["num_scored"], agg["mean_iou"], agg["median_iou"], agg["mean_auc"])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--tgif-dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--hybrid-fusion-checkpoint", type=str, default=None)
    parser.add_argument("--var-id", type=str, default=None, dest="var_id_filter")
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_tgif2_localization_eval_config(
        {
            "manifest_path": args.manifest_path,
            "tgif_dataset_dir": args.tgif_dataset_dir,
            "output_dir": args.output_dir,
            "hybrid_fusion_checkpoint": args.hybrid_fusion_checkpoint,
            "var_id_filter": args.var_id_filter,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / "tgif2_localization_eval.log")
    LOGGER.info("Loaded configuration (var_id_filter=%s)", config.var_id_filter)

    if not config.hybrid_fusion_checkpoint.exists():
        LOGGER.error("Hybrid fusion checkpoint not found: %s", config.hybrid_fusion_checkpoint)
        return 1
    if not config.manifest_path.exists():
        LOGGER.error("TGIF2 manifest not found: %s (run build_tgif2_manifest.py first)", config.manifest_path)
        return 1

    try:
        return Tgif2LocalizationRunner(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during TGIF2 localization evaluation:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
