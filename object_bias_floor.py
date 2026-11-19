"""Compute the object-bias floor for each ViT/CNN explanation method and floor-correct their
localization IoU on the 176 manipulated test images.

Per OBJECT_BIAS_FLOOR_SPEC.md: ground-truth masks are object-shaped, so an explanation method can
overlap a mask substantially just by finding the underlying object salient, whether or not it was
manipulated. This script scores each method's heatmap on the 176 REAL test images (Phase 6's
gradcam.py + this project's attention_rollout.py already generated heatmaps for all 352 test
images, real and fake) against the mask its FAKE counterpart uses - i.e. "the region that would
have been edited" - and takes floor(method) = median IoU over those 176 real images. DeltaIoU =
IoU - floor(method) is then reported for the 176 fake images, reusing Phase 7's
(localization_eval.py) already-computed cnn_cam/vit_cam/hybrid_cam IoU rather than recomputing it,
and scoring vit_rollout fresh (Phase 7 predates attention_rollout.py and never scored it).

Usage:
    python object_bias_floor.py
    python object_bias_floor.py --dry-run

Requires:
    pip install scikit-learn numpy pillow
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

from config import ObjectBiasFloorConfig, load_object_bias_floor_config
from localization_eval import LocalizationMetricsCalculator, MaskLoader
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("object_bias_floor")

METHODS = ("cnn_cam", "vit_cam", "hybrid_cam", "vit_rollout")


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


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


class ObjectBiasFloorCalculator:
    def __init__(self, config: ObjectBiasFloorConfig):
        self._config = config

    def run(self, dry_run: bool = False) -> int:
        config = self._config
        set_seed(config.seed)

        gradcam_rows = _read_csv(config.gradcam_dir / "gradcam_summary.csv")
        rollout_rows = _read_csv(config.attention_rollout_dir / "attention_rollout_summary.csv")
        per_image_rows = _read_csv(config.localization_eval_dir / "per_image_localization.csv")

        rollout_by_key = {(r["image_id"], r["y_true"]): r for r in rollout_rows}
        per_image_by_id = {r["image_id"]: r for r in per_image_rows}

        real_gradcam_rows = [r for r in gradcam_rows if int(r["y_true"]) == 0]
        fake_gradcam_rows = [r for r in gradcam_rows if int(r["y_true"]) == 1]
        if dry_run:
            real_gradcam_rows = real_gradcam_rows[:3]
            fake_gradcam_rows = fake_gradcam_rows[:3]
        LOGGER.info("Scoring %d real images to establish the object-bias floor", len(real_gradcam_rows))

        real_scores: dict[str, list[float]] = {m: [] for m in METHODS}
        real_score_rows: list[dict] = []
        skipped_real = 0

        for row in real_gradcam_rows:
            image_id, file_name = row["image_id"], row["file_name"]
            mask_path = config.data_dir / "masks" / f"{Path(file_name).stem}_mask.png"
            if not mask_path.exists():
                LOGGER.warning("No mask found for real image_id=%s (%s), skipping", image_id, file_name)
                skipped_real += 1
                continue
            mask = MaskLoader.load_binary_mask(mask_path)

            gradcam_raw = np.load(row["raw_npz_path"])
            rollout_row = rollout_by_key.get((image_id, row["y_true"]))
            if rollout_row is None:
                LOGGER.warning("No rollout heatmap found for real image_id=%s, skipping", image_id)
                skipped_real += 1
                continue
            rollout_raw = np.load(rollout_row["raw_npz_path"])

            heatmaps = {
                "cnn_cam": gradcam_raw["cnn_cam"], "vit_cam": gradcam_raw["vit_cam"],
                "hybrid_cam": gradcam_raw["hybrid_cam"], "vit_rollout": rollout_raw["vit_rollout"],
            }
            score_row = {"image_id": image_id, "file_name": file_name}
            any_scored = False
            for method in METHODS:
                metrics = LocalizationMetricsCalculator.compute(mask, heatmaps[method], config.binarization_threshold)
                if metrics is not None:
                    real_scores[method].append(metrics.iou)
                    any_scored = True
                    for field, value in metrics.to_dict().items():
                        score_row[f"{method}_{field}"] = value
            if any_scored:
                real_score_rows.append(score_row)
            else:
                skipped_real += 1

        floors = {
            method: (float(np.median(values)) if values else None)
            for method, values in real_scores.items()
        }
        for method, floor in floors.items():
            LOGGER.info("floor(%s) = %s (n=%d)", method, f"{floor:.4f}" if floor is not None else "n/a", len(real_scores[method]))

        LOGGER.info("Scoring %d fake images for DeltaIoU", len(fake_gradcam_rows))
        delta_rows: list[dict] = []
        skipped_fake = 0

        for row in fake_gradcam_rows:
            image_id, file_name = row["image_id"], row["file_name"]
            mask_path = config.data_dir / "masks" / f"{Path(file_name).stem}_mask.png"
            per_image_row = per_image_by_id.get(image_id)
            rollout_row = rollout_by_key.get((image_id, row["y_true"]))

            delta_row = {"image_id": image_id, "file_name": file_name}
            any_scored = False

            for method in ("cnn_cam", "vit_cam", "hybrid_cam"):
                if per_image_row is None or f"{method}_iou" not in per_image_row or per_image_row[f"{method}_iou"] == "":
                    continue
                iou = float(per_image_row[f"{method}_iou"])
                floor = floors[method]
                delta_row[f"{method}_iou"] = iou
                delta_row[f"{method}_floor"] = floor
                delta_row[f"{method}_delta_iou"] = iou - floor if floor is not None else None
                any_scored = True

            if mask_path.exists() and rollout_row is not None:
                mask = MaskLoader.load_binary_mask(mask_path)
                rollout_raw = np.load(rollout_row["raw_npz_path"])
                metrics = LocalizationMetricsCalculator.compute(mask, rollout_raw["vit_rollout"], config.binarization_threshold)
                if metrics is not None:
                    floor = floors["vit_rollout"]
                    delta_row["vit_rollout_iou"] = metrics.iou
                    delta_row["vit_rollout_floor"] = floor
                    delta_row["vit_rollout_delta_iou"] = metrics.iou - floor if floor is not None else None
                    any_scored = True

            if any_scored:
                delta_rows.append(delta_row)
            else:
                skipped_fake += 1

        self._write_floor_values(floors, len(real_gradcam_rows), skipped_real)
        self._write_real_image_scores(real_score_rows)
        self._write_delta_iou(delta_rows)

        if not dry_run:
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)

        LOGGER.info("Object-bias floor computation complete. Outputs written to %s", config.output_dir)
        return 0

    def _write_floor_values(self, floors: dict, num_real_candidates: int, num_skipped: int) -> None:
        payload = {"floor": floors, "num_real_images_scored": num_real_candidates - num_skipped, "num_skipped": num_skipped}
        path = self._config.output_dir / self._config.floor_values_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        LOGGER.info("Wrote floor values to %s", path)

    def _write_real_image_scores(self, rows: list[dict]) -> None:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        path = self._config.output_dir / self._config.real_image_scores_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote real-image scores for %d images to %s", len(rows), path)

    def _write_delta_iou(self, rows: list[dict]) -> None:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        path = self._config.output_dir / self._config.delta_iou_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote DeltaIoU for %d fake images to %s", len(rows), path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--gradcam-dir", type=str, default=None)
    parser.add_argument("--attention-rollout-dir", type=str, default=None)
    parser.add_argument("--localization-eval-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_object_bias_floor_config(
        {
            "data_dir": args.data_dir,
            "gradcam_dir": args.gradcam_dir,
            "attention_rollout_dir": args.attention_rollout_dir,
            "localization_eval_dir": args.localization_eval_dir,
            "output_dir": args.output_dir,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / "object_bias_floor.log")
    LOGGER.info("Loaded configuration (threshold=%.2f)", config.binarization_threshold)

    try:
        return ObjectBiasFloorCalculator(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during object-bias floor computation:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
