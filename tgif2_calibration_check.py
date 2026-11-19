"""TGIF2 score distribution check, per user's "TGIF2 Score Distribution Check" task (2026-09-16):
is the 92.2% zero-shot misclassification rate a representation problem (features don't separate
the classes) or a calibration problem (features separate, but the threshold doesn't transfer)?

Deviation, logged: the task says "authentic images (the 684 originals)," but TGIF2 actually has
1,368 unique authentic images (684 from tgif-images/orig/, shared by the tgif+random shares, plus
684 from tgif-flux/orig-flux/, used by the flux share) - established in this session's earlier
object-bias-floor work. Using all 1,368 (the full available authentic population) rather than an
arbitrary 684-image subset of it, since restricting to one pool would silently drop the flux
share's own authentic images and bias the per-share stratification the task also asks for.

No retraining, no new downloads - the authentic images already exist on disk (used earlier this
session for the object-bias floor); only a fresh classification-only forward pass (no Grad-CAM, no
backward pass) is run per authentic image, since their y_prob was never saved before now.
Manipulated-image y_prob is reused directly from tgif2_per_image_localization.csv (already
computed, not rerun).

Usage:
    python tgif2_calibration_check.py

Requires:
    pip install torch torchvision pandas numpy scikit-learn matplotlib
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from config import load_hybrid_fusion_config
from dataset import TransformFactory
from train_hybrid_fusion import HybridFusionClassifier

LOGGER = logging.getLogger("tgif2_calibration_check")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)

OUTPUT_DIR = Path("outputs/tgif2_calibration")
FLOOR_SCORES_PATH = Path("outputs/tgif2_object_bias_floor/tgif2_authentic_image_scores.csv")
PER_IMAGE_PATH = Path("outputs/tgif2_localization_eval/tgif2_per_image_localization.csv")
CHECKPOINT_PATH = Path("outputs/hybrid_fusion/best_model.pth")


def _score_authentic_images(model, transform, device) -> pd.DataFrame:
    floor_scores = pd.read_csv(FLOOR_SCORES_PATH)
    unique_images = floor_scores[["orig_path", "source_share"]].drop_duplicates("orig_path")
    LOGGER.info("Scoring %d unique authentic TGIF2 images (fresh forward pass, no Grad-CAM)", len(unique_images))

    rows = []
    for idx, row in enumerate(unique_images.itertuples()):
        with Image.open(row.orig_path) as img:
            image_tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(image_tensor).squeeze()
            y_prob = float(torch.sigmoid(logit).item())
        # tgif-images/orig/ is shared by the tgif AND random manipulated shares (established in
        # earlier object-bias-floor work); tgif-flux/orig-flux/ backs only the flux share.
        orig_pool = "tgif_images_orig" if "tgif-images/orig" in row.orig_path else "tgif_flux_orig"
        rows.append({"orig_path": row.orig_path, "orig_pool": orig_pool, "y_prob": y_prob, "y_true": 0})
        if (idx + 1) % 200 == 0 or (idx + 1) == len(unique_images):
            LOGGER.info("Scored %d/%d authentic images", idx + 1, len(unique_images))
    return pd.DataFrame(rows)


def _optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(y_prob)
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    return best_thresh, best_f1


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def _plot_distributions(manipulated_prob: np.ndarray, authentic_prob: np.ndarray, path: Path, title_suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(authentic_prob, bins=bins, alpha=0.6, label=f"Authentic (n={len(authentic_prob)})", color="#4C72B0", density=True)
    ax.hist(manipulated_prob, bins=bins, alpha=0.6, label=f"Manipulated (n={len(manipulated_prob)})", color="#C44E52", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Default threshold (0.5)")
    ax.set_xlabel("P(manipulated)")
    ax.set_ylabel("Density")
    ax.set_title(f"TGIF2 zero-shot score distributions{title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved distribution plot to %s", path)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    fusion_config = load_hybrid_fusion_config({})
    model = HybridFusionClassifier(fusion_config)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    transform = TransformFactory.build(fusion_config, train=False)

    authentic_df = _score_authentic_images(model, transform, device)
    authentic_df.to_csv(OUTPUT_DIR / "tgif2_authentic_y_prob.csv", index=False)

    manipulated_df = pd.read_csv(PER_IMAGE_PATH)[["coco_id", "source_share", "y_prob"]].copy()
    manipulated_df["y_true"] = 1

    LOGGER.info("Manipulated: n=%d, Authentic: n=%d", len(manipulated_df), len(authentic_df))

    # Pooled distribution plot + ROC-AUC + threshold analysis
    _plot_distributions(manipulated_df["y_prob"].values, authentic_df["y_prob"].values, OUTPUT_DIR / "distributions_pooled.png")

    combined_y_true = np.concatenate([manipulated_df["y_true"].values, authentic_df["y_true"].values])
    combined_y_prob = np.concatenate([manipulated_df["y_prob"].values, authentic_df["y_prob"].values])
    overall_auc = float(roc_auc_score(combined_y_true, combined_y_prob))
    LOGGER.info("Overall ROC-AUC (manipulated vs authentic, TGIF2 zero-shot): %.4f", overall_auc)

    optimal_threshold, optimal_f1 = _optimal_f1_threshold(combined_y_true, combined_y_prob)
    metrics_default = _metrics_at_threshold(combined_y_true, combined_y_prob, 0.5)
    metrics_optimal = _metrics_at_threshold(combined_y_true, combined_y_prob, optimal_threshold)
    LOGGER.info("At default threshold 0.5: %s", metrics_default)
    LOGGER.info("At TGIF2-optimal threshold %.4f (DIAGNOSTIC ONLY - tuned on this data, not a legitimate performance claim): %s",
                optimal_threshold, metrics_optimal)

    # Per-share stratification. tgif and random manipulated shares both compare against the
    # tgif_images_orig authentic pool (they share the same underlying authentic images); flux
    # compares against tgif_flux_orig.
    share_to_pool = {"tgif": "tgif_images_orig", "random": "tgif_images_orig", "flux": "tgif_flux_orig"}
    per_share_results = {}
    for share, pool in share_to_pool.items():
        manip_subset = manipulated_df[manipulated_df["source_share"] == share]
        auth_subset = authentic_df[authentic_df["orig_pool"] == pool]
        y_true = np.concatenate([np.ones(len(manip_subset)), np.zeros(len(auth_subset))])
        y_prob = np.concatenate([manip_subset["y_prob"].values, auth_subset["y_prob"].values])
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None
        per_share_results[share] = {
            "n_manipulated": int(len(manip_subset)), "n_authentic": int(len(auth_subset)),
            "auc": auc, "metrics_at_0.5": _metrics_at_threshold(y_true, y_prob, 0.5),
        }
        _plot_distributions(manip_subset["y_prob"].values, auth_subset["y_prob"].values,
                             OUTPUT_DIR / f"distributions_{share}.png", title_suffix=f" - {share} share")
        LOGGER.info("[%s] n_manip=%d n_auth=%d AUC=%s", share, len(manip_subset), len(auth_subset), auc)

    interpretation = (
        "CALIBRATION PROBLEM - features separate the classes reasonably well (AUC well above 0.5), "
        "but scores are shifted below the 0.5 decision threshold; the decision threshold does not "
        "transfer, even though the underlying representation does"
        if overall_auc > 0.65 else
        "REPRESENTATION FAILURE - AUC is close to 0.5 and distributions overlap; features genuinely "
        "do not separate the classes zero-shot, not just a threshold issue"
    )
    LOGGER.info("Interpretation: %s", interpretation)

    result = {
        "n_manipulated": int(len(manipulated_df)), "n_authentic": int(len(authentic_df)),
        "authentic_image_count_deviation_note": "used all 1,368 unique authentic images (684 tgif_images_orig + 684 tgif_flux_orig), not a 684-image subset - see module docstring",
        "overall_auc": overall_auc,
        "metrics_at_default_threshold_0.5": metrics_default,
        "optimal_threshold_diagnostic_only": {"threshold": optimal_threshold, "f1_at_optimal": optimal_f1, **metrics_optimal,
                                                "warning": "Tuned on this test data - a diagnostic, not a legitimate performance claim"},
        "per_share": per_share_results,
        "interpretation": interpretation,
    }
    (OUTPUT_DIR / "calibration_check_results.json").write_text(json.dumps(result, indent=2))
    LOGGER.info("Wrote %s", OUTPUT_DIR / "calibration_check_results.json")
    LOGGER.info("Calibration check complete. Outputs in %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
