"""Follow-up to Phase 7's threshold_sweep.csv (which covered hybrid_cam only): sweeps cnn_cam
too, so the Discussion chapter's claim that the CNN branch outperforms the hybrid heatmap at
*every* tested threshold (not just the fixed 0.5 used in the main results table) is verified
against real data rather than asserted. Reuses the same saved heatmaps/masks, no recomputation.

Usage:
    python threshold_sweep_comparison.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

from localization_eval import MaskLoader, LocalizationMetricsCalculator

LOGGER = logging.getLogger("threshold_sweep_comparison")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HEATMAP_TYPES = ("cnn_cam", "hybrid_cam")


def main() -> int:
    rows = [r for r in csv.DictReader(open("outputs/gradcam/gradcam_summary.csv")) if r["y_true"] == "1"]

    output_rows = []
    for heatmap_type in HEATMAP_TYPES:
        for threshold in THRESHOLDS:
            ious = []
            for row in rows:
                mask_path = Path("data/masks") / f"{Path(row['file_name']).stem}_mask.png"
                mask = MaskLoader.load_binary_mask(mask_path)
                raw = np.load(row["raw_npz_path"])
                metrics = LocalizationMetricsCalculator.compute(mask, raw[heatmap_type], threshold)
                if metrics is not None:
                    ious.append(metrics.iou)
            mean_iou = float(np.mean(ious))
            output_rows.append({"heatmap_type": heatmap_type, "threshold": threshold, "mean_iou": mean_iou, "n": len(ious)})
            LOGGER.info("heatmap_type=%s threshold=%.1f mean_iou=%.4f", heatmap_type, threshold, mean_iou)

    output_path = Path("outputs/experiments/threshold_sweep_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["heatmap_type", "threshold", "mean_iou", "n"])
        writer.writeheader()
        writer.writerows(output_rows)
    LOGGER.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
