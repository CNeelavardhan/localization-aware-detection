"""Compute the TGIF2-specific object-bias floor for each explanation method, stratified by mask
geometry (segm/bbox/random).

Per TGIF2_STATISTICAL_TREATMENT_SPEC.md Part A: unlike v1's object_bias_floor.py, no heatmaps exist
yet for TGIF2's authentic images (Phase 6-equivalent never ran on them), so this script generates
them fresh - one CNN/ViT forward+backward pass per unique authentic image (1,368 total: 684 from
tgif-images/orig/, shared by the tgif+random source shares, and 684 from tgif-flux/orig-flux/) -
then scores each authentic image's heatmap against every mask that exists for its coco_id at
var_id=0, per FINAL_PROJECT_METHODOLOGY.md Section 13's ground-truth-mask-shaped-bias rationale
(the same logic object_bias_floor.py already applies to v1). floor(method, mask_type) = median IoU
over all resulting (authentic image, mask_type) scores.

tgif2_localization_eval.py's Tgif2ManifestLoader is imported, not reimplemented, for orig/mask path
resolution. gradcam.py, attention_rollout.py, and localization_eval.py stay frozen.

Usage:
    python tgif2_object_bias_floor.py
    python tgif2_object_bias_floor.py --dry-run

Requires:
    pip install torch torchvision scikit-learn numpy pillow
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from attention_rollout import AttentionRolloutGenerator
from config import Tgif2ObjectBiasFloorConfig, load_hybrid_fusion_config, load_tgif2_object_bias_floor_config
from dataset import TransformFactory
from gradcam import GradCAMPlusPlus
from localization_eval import HEATMAP_TYPES as GRADCAM_HEATMAP_TYPES
from localization_eval import LocalizationMetricsCalculator, MaskLoader
from tgif2_localization_eval import Tgif2ManifestLoader
from train_hybrid_fusion import HybridFusionClassifier
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("tgif2_object_bias_floor")

ALL_HEATMAP_TYPES = (*GRADCAM_HEATMAP_TYPES, "vit_rollout")


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


class Tgif2ObjectBiasFloorCalculator:
    def __init__(self, config: Tgif2ObjectBiasFloorConfig):
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
            # Keep enough rows to exercise heatmap caching (repeat coco_ids) and multiple mask types.
            rows = rows[:20]

        config.output_dir.mkdir(parents=True, exist_ok=True)

        heatmap_cache: dict[str, dict] = {}
        score_rows: list[dict] = []
        scores_by_method_masktype: dict[tuple[str, str], list[float]] = {}
        unresolved = 0
        num_authentic_images = 0

        for idx, row in enumerate(rows):
            resolved = manifest_loader.resolve(row)
            if resolved is None:
                unresolved += 1
                LOGGER.warning("Could not resolve orig/mask for row: %s", row)
                continue
            orig_path, mask_path = resolved
            cache_key = str(orig_path)

            if cache_key not in heatmap_cache:
                with Image.open(orig_path) as img:
                    image_tensor = transform(img.convert("RGB")).unsqueeze(0)
                cam_result = cam_generator.generate(image_tensor)
                vit_rollout = rollout_generator.generate(image_tensor)
                heatmap_cache[cache_key] = {
                    "cnn_cam": cam_result["cnn_cam"], "vit_cam": cam_result["vit_cam"],
                    "hybrid_cam": cam_result["hybrid_cam"], "vit_rollout": vit_rollout,
                }
                num_authentic_images += 1
                if num_authentic_images % 200 == 0:
                    LOGGER.info("Computed heatmaps for %d unique authentic images so far", num_authentic_images)

            heatmaps = heatmap_cache[cache_key]
            mask = MaskLoader.load_binary_mask(mask_path)
            mask_type = row["mask_type"]

            score_row = {
                "orig_path": cache_key, "coco_id": row["coco_id"], "source_share": row["source_share"],
                "mask_type": mask_type, "split": row["split"],
            }
            any_scored = False
            for method in ALL_HEATMAP_TYPES:
                metrics = LocalizationMetricsCalculator.compute(mask, heatmaps[method], config.binarization_threshold)
                if metrics is not None:
                    score_row[f"{method}_iou"] = metrics.iou
                    scores_by_method_masktype.setdefault((method, mask_type), []).append(metrics.iou)
                    any_scored = True
            if any_scored:
                score_rows.append(score_row)

            if (idx + 1) % 500 == 0 or (idx + 1) == len(rows):
                LOGGER.info(
                    "Processed %d/%d manifest rows (%d unique authentic images, %d unresolved)",
                    idx + 1, len(rows), num_authentic_images, unresolved,
                )

        floors = self._compute_floors(scores_by_method_masktype)
        self._write_floor_values(floors, len(rows), num_authentic_images, unresolved)
        self._write_scores(score_rows)

        if not dry_run:
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)

        LOGGER.info("TGIF2 object-bias floor computation complete. Outputs written to %s", config.output_dir)
        return 0

    @staticmethod
    def _compute_floors(scores_by_method_masktype: dict[tuple[str, str], list[float]]) -> dict:
        mask_types = sorted({mt for (_method, mt) in scores_by_method_masktype})
        floors: dict = {}
        for method in ALL_HEATMAP_TYPES:
            floors[method] = {}
            for mask_type in mask_types:
                values = scores_by_method_masktype.get((method, mask_type), [])
                floors[method][mask_type] = {
                    "median_iou": float(np.median(values)) if values else None,
                    "num_scores": len(values),
                }
        return floors

    def _write_floor_values(self, floors: dict, num_rows: int, num_authentic_images: int, unresolved: int) -> None:
        payload = {
            "num_manifest_rows": num_rows, "num_unique_authentic_images": num_authentic_images,
            "num_unresolved": unresolved, "floor": floors,
        }
        path = self._config.output_dir / self._config.floor_values_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        LOGGER.info("Wrote floor values to %s", path)
        for method, by_mask_type in floors.items():
            for mask_type, stats in by_mask_type.items():
                if stats["median_iou"] is not None:
                    LOGGER.info(
                        "floor(%s, %s) = %.4f (n=%d)", method, mask_type, stats["median_iou"], stats["num_scores"],
                    )

    def _write_scores(self, rows: list[dict]) -> None:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        path = self._config.output_dir / self._config.authentic_image_scores_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote authentic-image scores for %d (image, mask_type) pairs to %s", len(rows), path)


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
    config = load_tgif2_object_bias_floor_config(
        {
            "manifest_path": args.manifest_path,
            "tgif_dataset_dir": args.tgif_dataset_dir,
            "output_dir": args.output_dir,
            "hybrid_fusion_checkpoint": args.hybrid_fusion_checkpoint,
            "var_id_filter": args.var_id_filter,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / "tgif2_object_bias_floor.log")
    LOGGER.info("Loaded configuration (var_id_filter=%s)", config.var_id_filter)

    if not config.hybrid_fusion_checkpoint.exists():
        LOGGER.error("Hybrid fusion checkpoint not found: %s", config.hybrid_fusion_checkpoint)
        return 1
    if not config.manifest_path.exists():
        LOGGER.error("TGIF2 manifest not found: %s (run build_tgif2_manifest.py first)", config.manifest_path)
        return 1

    try:
        return Tgif2ObjectBiasFloorCalculator(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during TGIF2 object-bias floor computation:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
