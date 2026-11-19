"""Phase 8 experiments: statistical significance, breakdowns, correlation, and efficiency
comparisons synthesized from Phases 3-7's already-computed outputs.

Per EXPERIMENTS_SPEC.md: trains nothing new and generates no new heatmaps. Each experiment reads
existing CSV/JSON outputs (test_predictions.csv, per_image_localization.csv, gradcam_summary.csv)
or, for E4 only, loads the already-trained checkpoints purely to count parameters and time
inference (no gradient computation, no new predictions saved).

Usage:
    python experiments.py                      # runs all four experiments
    python experiments.py --experiments e1,e3   # runs only E1 and E3

Requires:
    pip install numpy scipy matplotlib torch torchvision
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

LOGGER = logging.getLogger("experiments")

SEED = 42


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


def _read_predictions(path: Path) -> dict[tuple[str, str], dict]:
    with path.open("r", newline="") as f:
        return {(row["image_id"], row["y_true"]): row for row in csv.DictReader(f)}


class SignificanceTester:
    """E1: McNemar's test comparing the Hybrid model's correctness against each standalone
    backbone's, on the identical 352 test images.
    """

    @staticmethod
    def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
        correct_a = pred_a == y_true
        correct_b = pred_b == y_true
        n01 = int(np.sum(correct_a & ~correct_b))
        n10 = int(np.sum(~correct_a & correct_b))
        n = n01 + n10

        if n == 0:
            return {"n_a_correct_b_wrong": n01, "n_a_wrong_b_correct": n10, "statistic": None, "p_value": 1.0, "method": "degenerate"}

        if n < 25:
            from scipy.stats import binomtest

            result = binomtest(min(n01, n10), n, 0.5)
            return {"n_a_correct_b_wrong": n01, "n_a_wrong_b_correct": n10, "statistic": None, "p_value": float(result.pvalue), "method": "exact_binomial"}

        from scipy.stats import chi2

        statistic = (abs(n01 - n10) - 1) ** 2 / n
        p_value = float(1 - chi2.cdf(statistic, df=1))
        return {"n_a_correct_b_wrong": n01, "n_a_wrong_b_correct": n10, "statistic": float(statistic), "p_value": p_value, "method": "chi2_continuity_corrected"}

    def run(self, output_dir: Path) -> None:
        hybrid = _read_predictions(Path("outputs/hybrid_fusion/test_predictions.csv"))
        efficientnet = _read_predictions(Path("outputs/efficientnet_b3/test_predictions.csv"))
        vit = _read_predictions(Path("outputs/vit_b16/test_predictions.csv"))

        keys = sorted(set(hybrid) & set(efficientnet) & set(vit))
        y_true = np.array([int(k[1]) for k in keys])
        hybrid_pred = np.array([int(hybrid[k]["y_pred"]) for k in keys])
        efficientnet_pred = np.array([int(efficientnet[k]["y_pred"]) for k in keys])
        vit_pred = np.array([int(vit[k]["y_pred"]) for k in keys])

        results = {
            "num_samples": len(keys),
            "hybrid_vs_efficientnet_b3": self.mcnemar_test(y_true, hybrid_pred, efficientnet_pred),
            "hybrid_vs_vit_b16": self.mcnemar_test(y_true, hybrid_pred, vit_pred),
        }
        for comparison, result in results.items():
            if comparison != "num_samples":
                LOGGER.info("[E1] %s: %s", comparison, result)

        path = output_dir / "mcnemar_test_results.json"
        path.write_text(json.dumps(results, indent=2))
        LOGGER.info("[E1] Wrote %s", path)


class LocalizationBreakdown:
    """E2: groups per_image_localization.csv's cnn_cam metrics by object_class, manipulation_type,
    and mask_area_ratio tertile.
    """

    def run(self, output_dir: Path) -> None:
        rows = self._read_rows()
        self._write_breakdown(rows, "object_class", output_dir / "localization_breakdown_by_object_class.csv")
        self._write_breakdown(rows, "manipulation_type", output_dir / "localization_breakdown_by_manipulation_type.csv")
        self._write_area_breakdown(rows, output_dir / "localization_breakdown_by_mask_area.csv")

    @staticmethod
    def _read_rows() -> list[dict]:
        path = Path("outputs/localization_eval/per_image_localization.csv")
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _write_breakdown(rows: list[dict], group_field: str, output_path: Path) -> None:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            key = row.get(group_field, "") or "unknown"
            groups.setdefault(key, []).append(row)

        breakdown_rows = []
        for key, group_rows in sorted(groups.items()):
            ious = [float(r["cnn_cam_iou"]) for r in group_rows if r.get("cnn_cam_iou")]
            dices = [float(r["cnn_cam_dice"]) for r in group_rows if r.get("cnn_cam_dice")]
            aucs = [float(r["cnn_cam_auc"]) for r in group_rows if r.get("cnn_cam_auc")]
            if not ious:
                continue
            breakdown_rows.append(
                {
                    group_field: key, "n": len(ious),
                    "mean_cnn_cam_iou": float(np.mean(ious)), "mean_cnn_cam_dice": float(np.mean(dices)),
                    "mean_cnn_cam_auc": float(np.mean(aucs)),
                }
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[group_field, "n", "mean_cnn_cam_iou", "mean_cnn_cam_dice", "mean_cnn_cam_auc"])
            writer.writeheader()
            writer.writerows(breakdown_rows)
        LOGGER.info("[E2] Wrote %s (%d groups)", output_path, len(breakdown_rows))

    @staticmethod
    def _write_area_breakdown(rows: list[dict], output_path: Path) -> None:
        valid_rows = [r for r in rows if r.get("mask_area_ratio") and r.get("cnn_cam_iou")]
        areas = np.array([float(r["mask_area_ratio"]) for r in valid_rows])
        if len(areas) == 0:
            return
        p33, p66 = np.percentile(areas, [33.3, 66.7])

        def bucket(area: float) -> str:
            if area <= p33:
                return "small"
            if area <= p66:
                return "medium"
            return "large"

        groups: dict[str, list[dict]] = {"small": [], "medium": [], "large": []}
        for row, area in zip(valid_rows, areas):
            groups[bucket(area)].append(row)

        breakdown_rows = []
        for key in ("small", "medium", "large"):
            group_rows = groups[key]
            if not group_rows:
                continue
            ious = [float(r["cnn_cam_iou"]) for r in group_rows]
            dices = [float(r["cnn_cam_dice"]) for r in group_rows]
            aucs = [float(r["cnn_cam_auc"]) for r in group_rows]
            area_values = [float(r["mask_area_ratio"]) for r in group_rows]
            breakdown_rows.append(
                {
                    "size_bucket": key, "n": len(ious),
                    "mask_area_ratio_range": f"{min(area_values):.4f}-{max(area_values):.4f}",
                    "mean_cnn_cam_iou": float(np.mean(ious)), "mean_cnn_cam_dice": float(np.mean(dices)),
                    "mean_cnn_cam_auc": float(np.mean(aucs)),
                }
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["size_bucket", "n", "mask_area_ratio_range", "mean_cnn_cam_iou", "mean_cnn_cam_dice", "mean_cnn_cam_auc"])
            writer.writeheader()
            writer.writerows(breakdown_rows)
        LOGGER.info("[E2] Wrote %s (tertile cutoffs: p33=%.4f, p66=%.4f)", output_path, p33, p66)


class ConfidenceCorrelation:
    """E3: correlates the fused classifier's confidence (y_prob) with cnn_cam localization IoU,
    for the 176 manipulated test images.
    """

    def run(self, output_dir: Path) -> None:
        confidence_by_image = self._read_confidence()
        localization_rows = self._read_localization()

        joined = []
        for row in localization_rows:
            image_id = row["image_id"]
            if image_id in confidence_by_image and row.get("cnn_cam_iou"):
                joined.append({"image_id": image_id, "y_prob": confidence_by_image[image_id], "cnn_cam_iou": float(row["cnn_cam_iou"])})

        if len(joined) < 2:
            LOGGER.warning("[E3] Not enough joined samples to compute correlation")
            return

        y_prob = np.array([r["y_prob"] for r in joined])
        iou = np.array([r["cnn_cam_iou"] for r in joined])

        from scipy.stats import pearsonr

        correlation, p_value = pearsonr(y_prob, iou)
        LOGGER.info("[E3] Pearson correlation(y_prob, cnn_cam_iou) = %.4f (p=%.4f, n=%d)", correlation, p_value, len(joined))

        csv_path = output_dir / "confidence_vs_localization.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "y_prob", "cnn_cam_iou"])
            writer.writeheader()
            writer.writerows(joined)

        self._save_scatter(y_prob, iou, correlation, p_value, output_dir / "confidence_vs_localization_scatter.png")
        LOGGER.info("[E3] Wrote %s and scatter plot", csv_path)

    @staticmethod
    def _read_confidence() -> dict[str, float]:
        path = Path("outputs/gradcam/gradcam_summary.csv")
        with path.open("r", newline="") as f:
            return {row["image_id"]: float(row["y_prob"]) for row in csv.DictReader(f) if row["y_true"] == "1"}

    @staticmethod
    def _read_localization() -> list[dict]:
        path = Path("outputs/localization_eval/per_image_localization.csv")
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _save_scatter(y_prob: np.ndarray, iou: np.ndarray, correlation: float, p_value: float, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_prob, iou, alpha=0.6)
        ax.set_xlabel("Fused classifier confidence (y_prob)")
        ax.set_ylabel("CNN-branch Grad-CAM++ IoU")
        ax.set_title(f"Confidence vs. Localization (r={correlation:.3f}, p={p_value:.4f})")

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


class EfficiencyBenchmark:
    """E4: parameter counts and measured CPU inference latency for all three trained models."""

    def run(self, output_dir: Path) -> None:
        import torch

        from config import load_config, load_hybrid_fusion_config, load_vit_b16_config
        from train_efficientnet_b3 import EfficientNetB3Classifier
        from train_hybrid_fusion import HybridFusionClassifier
        from train_vit_b16 import ViTB16Classifier

        device = torch.device("cpu")
        rows = []

        efficientnet_config = load_config()
        efficientnet_model = EfficientNetB3Classifier(dropout=efficientnet_config.dropout, dense_units=efficientnet_config.dense_units)
        efficientnet_model.load_state_dict(torch.load(efficientnet_config.output_dir / efficientnet_config.best_model_filename, map_location=device))
        rows.append(self._benchmark("EfficientNet-B3", efficientnet_model, device))

        vit_config = load_vit_b16_config()
        vit_model = ViTB16Classifier(dropout=vit_config.dropout, dense_units=vit_config.dense_units)
        vit_model.load_state_dict(torch.load(vit_config.output_dir / vit_config.best_model_filename, map_location=device))
        rows.append(self._benchmark("ViT-B/16", vit_model, device))

        fusion_config = load_hybrid_fusion_config()
        fusion_model = HybridFusionClassifier(fusion_config)
        fusion_model.load_state_dict(torch.load(fusion_config.output_dir / fusion_config.best_model_filename, map_location=device))
        rows.append(self._benchmark("Hybrid Fusion", fusion_model, device))

        path = output_dir / "efficiency_comparison.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            LOGGER.info("[E4] %s", row)
        LOGGER.info("[E4] Wrote %s", path)

    @staticmethod
    def _benchmark(name: str, model, device, num_images: int = 20) -> dict:
        import torch

        model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            for _ in range(3):  # warm-up
                model(dummy)

        start = time.time()
        with torch.no_grad():
            for _ in range(num_images):
                model(dummy)
        elapsed = time.time() - start

        return {
            "model": name, "total_params": total_params, "trainable_params": trainable_params,
            "mean_inference_latency_ms": round((elapsed / num_images) * 1000, 2),
        }


EXPERIMENTS = {
    "e1": ("SignificanceTester", SignificanceTester),
    "e2": ("LocalizationBreakdown", LocalizationBreakdown),
    "e3": ("ConfidenceCorrelation", ConfidenceCorrelation),
    "e4": ("EfficiencyBenchmark", EfficiencyBenchmark),
}


def _write_reproducibility(output_dir: Path, requested: list[str]) -> None:
    import platform

    import torch
    import torchvision

    info = {
        "seed": SEED,
        "experiments_run": requested,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    path = output_dir / "reproducibility.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
    LOGGER.info("Wrote %s", path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=str, default="e1,e2,e3,e4")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    configure_logging(args.log_level, output_dir / "experiments.log")

    requested = [e.strip().lower() for e in args.experiments.split(",")]
    unknown = [e for e in requested if e not in EXPERIMENTS]
    if unknown:
        LOGGER.error("Unknown experiment(s): %s (valid: %s)", unknown, list(EXPERIMENTS))
        return 1

    try:
        for key in requested:
            name, experiment_cls = EXPERIMENTS[key]
            LOGGER.info("Running %s (%s)...", key.upper(), name)
            experiment_cls().run(output_dir)
        _write_reproducibility(output_dir, requested)
        LOGGER.info("All requested experiments complete. Outputs written to %s", output_dir)
        return 0
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during experiments:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
