"""Weighted-fusion ablation: does biasing the hybrid heatmap toward the CNN branch improve
localization over the naive 50/50 average found in Phase 7 (LOCALIZATION_EVAL_SPEC.md)?

Tests H = w_cnn * cnn_cam + (1 - w_cnn) * vit_cam at a sweep of weights, reusing
outputs/gradcam/test/*_raw.npz (the per-branch heatmaps Phase 6 already computed - no Grad-CAM++
recomputation needed) and localization_eval.py's MaskLoader/LocalizationMetricsCalculator (same
mask resizing, same fixed threshold=0.5) for direct comparability with Phase 7's reported numbers.
w_cnn=1.0 reproduces Phase 7's cnn_cam numbers exactly; w_cnn=0.5 reproduces its hybrid_cam numbers
exactly - both serve as a correctness check on this script.

Usage:
    python weighted_fusion_ablation.py
    python weighted_fusion_ablation.py --weights 1.0,0.9,0.8,0.7,0.6,0.5
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from config import load_gradcam_config, load_localization_eval_config
from localization_eval import LocalizationMetricsCalculator, MaskLoader

LOGGER = logging.getLogger("weighted_fusion_ablation")


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    LOGGER.addHandler(handler)


def run(gradcam_dir: Path, data_dir: Path, output_path: Path, weights: list[float], threshold: float) -> None:
    summary_path = gradcam_dir / "gradcam_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Grad-CAM++ summary not found: {summary_path} (run gradcam.py first)")
    with summary_path.open("r", newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["y_true"]) == 1]
    LOGGER.info("Loaded %d manipulated test images for ablation", len(rows))

    results: list[dict] = []
    for w_cnn in weights:
        metrics_list = []
        for row in rows:
            mask_path = data_dir / "masks" / f"{Path(row['file_name']).stem}_mask.png"
            if not mask_path.exists():
                continue
            mask = MaskLoader.load_binary_mask(mask_path)
            raw = np.load(row["raw_npz_path"])
            combined = w_cnn * raw["cnn_cam"] + (1.0 - w_cnn) * raw["vit_cam"]
            metrics = LocalizationMetricsCalculator.compute(mask, combined, threshold)
            if metrics is not None:
                metrics_list.append(metrics)

        if not metrics_list:
            continue

        result = {
            "cnn_weight": w_cnn,
            "vit_weight": round(1.0 - w_cnn, 2),
            "n": len(metrics_list),
            "mean_auc": float(np.mean([m.auc for m in metrics_list])),
            "mean_iou": float(np.mean([m.iou for m in metrics_list])),
            "mean_dice": float(np.mean([m.dice for m in metrics_list])),
            "mean_precision": float(np.mean([m.precision for m in metrics_list])),
            "mean_recall": float(np.mean([m.recall for m in metrics_list])),
        }
        results.append(result)
        LOGGER.info(
            "cnn_weight=%.2f vit_weight=%.2f | n=%d mean_auc=%.4f mean_iou=%.4f mean_dice=%.4f "
            "mean_precision=%.4f mean_recall=%.4f",
            result["cnn_weight"], result["vit_weight"], result["n"], result["mean_auc"],
            result["mean_iou"], result["mean_dice"], result["mean_precision"], result["mean_recall"],
        )

    if not results:
        LOGGER.error("No results produced (no scoreable images found)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    LOGGER.info("Wrote weighted-fusion ablation results to %s", output_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--gradcam-dir", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weights", type=str, default="1.0,0.9,0.8,0.7,0.6,0.5")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    gradcam_config = load_gradcam_config({"data_dir": args.data_dir, "output_dir": args.gradcam_dir})
    localization_config = load_localization_eval_config({"data_dir": args.data_dir})

    gradcam_dir = Path(args.gradcam_dir) if args.gradcam_dir else gradcam_config.output_dir
    data_dir = Path(args.data_dir) if args.data_dir else gradcam_config.data_dir
    output_path = Path(args.output_path) if args.output_path else Path("outputs/experiments/weighted_fusion_ablation.csv")
    weights = [float(w) for w in args.weights.split(",")]

    run(gradcam_dir, data_dir, output_path, weights, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
