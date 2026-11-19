"""Score Phase 6's Grad-CAM++ heatmaps against ground-truth manipulation masks.

Per LOCALIZATION_EVAL_SPEC.md: loads outputs/gradcam/test/*_raw.npz (already generated, not
re-run here) for the 176 manipulated test images, resizes each image's data/masks/*_mask.png down
to 224x224 (nearest-neighbor, matching the heatmaps' resolution), and computes pixel-level ROC-AUC
(threshold-free) plus IoU/Dice/Precision/Recall (at a fixed 0.5 threshold) for each of the three
heatmap types (cnn_cam, vit_cam, hybrid_cam). Also writes a threshold sweep and a handful of
best/worst-localized qualitative comparison figures.

Usage:
    python localization_eval.py
    python localization_eval.py --data-dir data --gradcam-dir outputs/gradcam --output-dir outputs/localization_eval
    python localization_eval.py --dry-run

Requires:
    pip install scikit-learn matplotlib pillow numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config import LocalizationEvalConfig, load_localization_eval_config
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("localization_eval")

HEATMAP_TYPES = ("cnn_cam", "vit_cam", "hybrid_cam")


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


class MaskLoader:
    @staticmethod
    def load_binary_mask(mask_path: Path, size: int = 224) -> np.ndarray:
        with Image.open(mask_path) as img:
            resized = img.convert("L").resize((size, size), Image.NEAREST)
            arr = np.array(resized)
        return arr >= 128


@dataclass(frozen=True)
class LocalizationMetrics:
    auc: float
    iou: float
    dice: float
    precision: float
    recall: float

    def to_dict(self) -> dict:
        return asdict(self)


class LocalizationMetricsCalculator:
    @staticmethod
    def compute(mask: np.ndarray, heatmap: np.ndarray, threshold: float) -> Optional[LocalizationMetrics]:
        y_true = mask.flatten().astype(int)
        y_prob = heatmap.flatten().astype(float)

        if len(set(y_true.tolist())) < 2:
            return None

        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_prob))

        y_pred = (y_prob >= threshold).astype(int)
        intersection = int(np.logical_and(y_true, y_pred).sum())
        union = int(np.logical_or(y_true, y_pred).sum())
        pred_sum = int(y_pred.sum())
        true_sum = int(y_true.sum())

        iou = intersection / union if union > 0 else 0.0
        dice = (2 * intersection) / (pred_sum + true_sum) if (pred_sum + true_sum) > 0 else 0.0
        precision = intersection / pred_sum if pred_sum > 0 else 0.0
        recall = intersection / true_sum if true_sum > 0 else 0.0

        return LocalizationMetrics(auc=auc, iou=iou, dice=dice, precision=precision, recall=recall)


class QualitativeFigureWriter:
    @staticmethod
    def save_comparison(original_image: Image.Image, mask: np.ndarray, heatmap: np.ndarray, path: Path, title: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        base = np.array(original_image.resize((224, 224))).astype(np.float64) / 255.0
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(base)
        axes[0].set_title("Manipulated Image")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        axes[2].imshow(base)
        axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[2].set_title("Hybrid CAM Overlay")
        axes[2].axis("off")
        fig.suptitle(title)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


class LocalizationEvaluator:
    def __init__(self, config: LocalizationEvalConfig):
        self._config = config

    def run(self, dry_run: bool = False) -> int:
        config = self._config
        set_seed(config.seed)

        summary_rows = self._read_gradcam_summary()
        manipulated_rows = [r for r in summary_rows if int(r["y_true"]) == 1]
        if dry_run:
            manipulated_rows = manipulated_rows[:3]
        LOGGER.info("Scoring %d manipulated test images against ground-truth masks", len(manipulated_rows))

        per_image_rows: list[dict] = []
        per_type_metrics: dict[str, list[LocalizationMetrics]] = {t: [] for t in HEATMAP_TYPES}
        skipped = 0

        for row in manipulated_rows:
            image_id = row["image_id"]
            file_name = row["file_name"]
            mask_path = config.data_dir / "masks" / f"{Path(file_name).stem}_mask.png"
            if not mask_path.exists():
                LOGGER.warning("No mask found for image_id=%s (%s), skipping", image_id, file_name)
                skipped += 1
                continue

            mask = MaskLoader.load_binary_mask(mask_path)
            raw = np.load(row["raw_npz_path"])

            metrics_by_type: dict[str, Optional[LocalizationMetrics]] = {}
            for heatmap_type in HEATMAP_TYPES:
                metrics = LocalizationMetricsCalculator.compute(mask, raw[heatmap_type], config.binarization_threshold)
                metrics_by_type[heatmap_type] = metrics
                if metrics is not None:
                    per_type_metrics[heatmap_type].append(metrics)

            if all(m is None for m in metrics_by_type.values()):
                LOGGER.warning("Degenerate mask after resize for image_id=%s, skipping", image_id)
                skipped += 1
                continue

            metadata = self._load_metadata(file_name)
            per_image_rows.append(self._build_row(image_id, file_name, metadata, metrics_by_type))

        self._write_metrics_json(per_type_metrics, len(manipulated_rows), skipped)
        self._write_per_image_csv(per_image_rows)

        if not dry_run:
            self._write_threshold_sweep(manipulated_rows)
            self._write_qualitative_examples(per_image_rows)
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)

        LOGGER.info("Localization evaluation complete. Outputs written to %s", config.output_dir)
        return 0

    def _read_gradcam_summary(self) -> list[dict]:
        summary_path = self._config.gradcam_dir / "gradcam_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Grad-CAM++ summary not found: {summary_path} (run gradcam.py first)")
        with summary_path.open("r", newline="") as f:
            return list(csv.DictReader(f))

    def _load_metadata(self, file_name: str) -> dict:
        metadata_path = self._config.data_dir / "metadata" / f"{Path(file_name).stem}.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text())

    @staticmethod
    def _build_row(image_id: str, file_name: str, metadata: dict, metrics_by_type: dict) -> dict:
        row = {
            "image_id": image_id,
            "file_name": file_name,
            "object_class": metadata.get("object_class", ""),
            "manipulation_type": metadata.get("manipulation_type", ""),
            "mask_area_ratio": metadata.get("mask_area_ratio", ""),
        }
        for heatmap_type in HEATMAP_TYPES:
            metrics = metrics_by_type[heatmap_type]
            for field_name, value in (metrics.to_dict().items() if metrics is not None else []):
                row[f"{heatmap_type}_{field_name}"] = value
        return row

    def _write_metrics_json(self, per_type_metrics: dict[str, list[LocalizationMetrics]], num_candidates: int, skipped: int) -> None:
        aggregate: dict = {"num_candidate_images": num_candidates, "num_skipped": skipped}
        for heatmap_type, metrics_list in per_type_metrics.items():
            if not metrics_list:
                aggregate[heatmap_type] = {"num_scored": 0}
                continue
            aggregate[heatmap_type] = {
                "num_scored": len(metrics_list),
                "mean_auc": float(np.mean([m.auc for m in metrics_list])),
                "mean_iou": float(np.mean([m.iou for m in metrics_list])),
                "mean_dice": float(np.mean([m.dice for m in metrics_list])),
                "mean_precision": float(np.mean([m.precision for m in metrics_list])),
                "mean_recall": float(np.mean([m.recall for m in metrics_list])),
            }
            LOGGER.info(
                "[%s] n=%d mean_auc=%.4f mean_iou=%.4f mean_dice=%.4f mean_precision=%.4f mean_recall=%.4f",
                heatmap_type, len(metrics_list), aggregate[heatmap_type]["mean_auc"],
                aggregate[heatmap_type]["mean_iou"], aggregate[heatmap_type]["mean_dice"],
                aggregate[heatmap_type]["mean_precision"], aggregate[heatmap_type]["mean_recall"],
            )

        path = self._config.output_dir / self._config.metrics_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2))

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

    def _write_threshold_sweep(self, manipulated_rows: list[dict]) -> None:
        sweep_rows = []
        for threshold in self._config.threshold_sweep:
            metrics_list = []
            for row in manipulated_rows:
                mask_path = self._config.data_dir / "masks" / f"{Path(row['file_name']).stem}_mask.png"
                if not mask_path.exists():
                    continue
                mask = MaskLoader.load_binary_mask(mask_path)
                raw = np.load(row["raw_npz_path"])
                metrics = LocalizationMetricsCalculator.compute(mask, raw["hybrid_cam"], threshold)
                if metrics is not None:
                    metrics_list.append(metrics)
            if metrics_list:
                sweep_rows.append(
                    {
                        "threshold": threshold,
                        "mean_iou": float(np.mean([m.iou for m in metrics_list])),
                        "mean_dice": float(np.mean([m.dice for m in metrics_list])),
                        "mean_precision": float(np.mean([m.precision for m in metrics_list])),
                        "mean_recall": float(np.mean([m.recall for m in metrics_list])),
                    }
                )

        path = self._config.output_dir / self._config.threshold_sweep_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["threshold", "mean_iou", "mean_dice", "mean_precision", "mean_recall"])
            writer.writeheader()
            writer.writerows(sweep_rows)
        LOGGER.info("Wrote threshold sweep (hybrid_cam) to %s", path)

    def _write_qualitative_examples(self, per_image_rows: list[dict]) -> None:
        scored = [r for r in per_image_rows if "hybrid_cam_iou" in r]
        if not scored:
            return
        scored_sorted = sorted(scored, key=lambda r: r["hybrid_cam_iou"])
        n = self._config.num_qualitative_examples
        worst = scored_sorted[:n]
        best = list(reversed(scored_sorted[-n:]))

        qual_dir = self._config.output_dir / self._config.qualitative_dirname
        for rank, row in enumerate(best, start=1):
            self._save_one_example(row, qual_dir / f"best_{rank}_{row['image_id']}.png", f"Best #{rank} (IoU={row['hybrid_cam_iou']:.3f})")
        for rank, row in enumerate(worst, start=1):
            self._save_one_example(row, qual_dir / f"worst_{rank}_{row['image_id']}.png", f"Worst #{rank} (IoU={row['hybrid_cam_iou']:.3f})")
        LOGGER.info("Wrote %d qualitative example figures to %s", 2 * n, qual_dir)

    def _save_one_example(self, row: dict, path: Path, title: str) -> None:
        image_id = row["image_id"]
        label_name = "fake"
        raw_path = self._config.gradcam_dir / "test" / f"{image_id}_{label_name}_raw.npz"
        raw = np.load(raw_path)
        mask_path = self._config.data_dir / "masks" / f"{Path(row['file_name']).stem}_mask.png"
        mask = MaskLoader.load_binary_mask(mask_path)
        manipulated_image = Image.open(self._config.data_dir / "manipulated" / row["file_name"]).convert("RGB")
        QualitativeFigureWriter.save_comparison(manipulated_image, mask, raw["hybrid_cam"], path, title)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--gradcam-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_localization_eval_config(
        {
            "data_dir": args.data_dir,
            "gradcam_dir": args.gradcam_dir,
            "output_dir": args.output_dir,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / "localization_eval.log")
    LOGGER.info("Loaded configuration (threshold=%.2f)", config.binarization_threshold)

    try:
        return LocalizationEvaluator(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during localization evaluation:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
