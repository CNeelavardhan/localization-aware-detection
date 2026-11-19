"""Fix 1 (dissociation check) and Fix 2 (random-mask power analysis), per
ADDITIONAL_ANALYSES_SPEC.md - implements the user's "Three Fixable Weaknesses" brief (2026-09-16).

Pure post-hoc reanalysis of already-computed TGIF2 outputs (tgif2_per_image_localization.csv,
tgif2_floor_values.json, tgif2_statistical_results.json) - no model, no new heatmaps, no torch
dependency, no retraining. Fix 3 (external IoU baseline) is documentation-only and not implemented
here.

Usage:
    python additional_analyses.py

Requires:
    pip install pandas numpy scipy
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

LOGGER = logging.getLogger("additional_analyses")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)

PER_IMAGE_PATH = Path("outputs/tgif2_localization_eval/tgif2_per_image_localization.csv")
FLOOR_PATH = Path("outputs/tgif2_object_bias_floor/tgif2_floor_values.json")
OUTPUT_DIR = Path("outputs/additional_analyses")
SEED = 42
N_RESAMPLES = 10000
CI_LEVEL = 0.95


def _load_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(PER_IMAGE_PATH)
    floors = json.loads(FLOOR_PATH.read_text())["floor"]
    for method in ("cnn_cam", "vit_rollout"):
        df[f"{method}_floor"] = df["mask_type"].map(lambda mt, m=method: floors[m][mt]["median_iou"])
        df[f"{method}_delta_iou"] = df[f"{method}_iou"] - df[f"{method}_floor"]
    return df, floors


def _bca_bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_resamples: int = N_RESAMPLES) -> tuple[float, float]:
    from scipy.stats import bootstrap

    if len(values) < 2 or np.allclose(values, values[0]):
        return (float(values.mean()), float(values.mean())) if len(values) else (float("nan"), float("nan"))
    result = bootstrap((values,), np.mean, n_resamples=n_resamples, confidence_level=CI_LEVEL, method="BCa", random_state=rng)
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def _cluster_bootstrap_group_mean(rows: pd.DataFrame, coco_ids: list, column: str, rng: np.random.Generator) -> tuple[float, float]:
    """Resamples coco_ids with replacement; each resample's statistic is the mean of `column`
    over all of the resampled coco_ids' rows in this group (a coco_id with none contributes
    nothing that resample - standard practice for unbalanced cluster bootstrap, per spec).
    """
    by_coco = {cid: g[column].values for cid, g in rows.groupby("coco_id")}
    coco_array = np.array(coco_ids)
    stats = []
    for _ in range(N_RESAMPLES):
        sample_ids = rng.choice(coco_array, size=len(coco_array), replace=True)
        pooled = np.concatenate([by_coco[cid] for cid in sample_ids if cid in by_coco])
        if len(pooled):
            stats.append(pooled.mean())
    stats = np.array(stats)
    if len(stats) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    return float(lo), float(hi)


def fix1_dissociation_check(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    LOGGER.info("=== Fix 1: Dissociation check ===")
    df = df.copy()
    df["correct"] = df["y_prob"] >= 0.5
    LOGGER.info("Row-level split: %d correctly classified, %d misclassified (of %d)",
                df["correct"].sum(), (~df["correct"]).sum(), len(df))

    result: dict = {}
    group_deltas_rows = []
    coco_level: dict[str, dict] = {}

    for group_name, is_correct in (("correctly_classified", True), ("misclassified", False)):
        group_rows = df[df["correct"] == is_correct]
        collapsed = group_rows.groupby("coco_id")["cnn_cam_delta_iou"].mean()
        coco_level[group_name] = collapsed
        for coco_id, value in collapsed.items():
            group_deltas_rows.append({"coco_id": coco_id, "group": group_name, "mean_delta_iou": value})

        values = collapsed.values
        n = len(values)
        if n >= 2 and not np.allclose(values, 0.0):
            stat, p = wilcoxon(values, alternative="two-sided")
        else:
            stat, p = None, None
        ci_low, ci_high = _bca_bootstrap_mean(values, rng)

        result[group_name] = {
            "n_coco_ids": int(n), "n_rows": int(len(group_rows)),
            "mean_delta_iou": float(np.mean(values)) if n else None,
            "median_delta_iou": float(np.median(values)) if n else None,
            "wilcoxon_vs_zero_statistic": float(stat) if stat is not None else None,
            "wilcoxon_vs_zero_p_value": float(p) if p is not None else None,
            "ci_95_bca": [ci_low, ci_high],
        }
        LOGGER.info("[%s] n_coco_ids=%d mean=%.4f median=%.4f p_vs_zero=%s ci=[%.4f, %.4f]",
                    group_name, n, result[group_name]["mean_delta_iou"], result[group_name]["median_delta_iou"],
                    result[group_name]["wilcoxon_vs_zero_p_value"], ci_low, ci_high)

    correct_vals = coco_level["correctly_classified"].values
    misclass_vals = coco_level["misclassified"].values
    if len(correct_vals) >= 2 and len(misclass_vals) >= 2:
        mw_stat, mw_p = mannwhitneyu(correct_vals, misclass_vals, alternative="two-sided")
        result["between_groups_mannwhitney"] = {"statistic": float(mw_stat), "p_value": float(mw_p)}
    LOGGER.info("Between-groups Mann-Whitney: %s", result.get("between_groups_mannwhitney"))

    # Cluster bootstrap CI on coco_id (spec's explicit requirement, distinct from the per-image BCa above)
    for group_name, is_correct in (("correctly_classified", True), ("misclassified", False)):
        rows = df[df["correct"] == is_correct]
        coco_ids = sorted(rows["coco_id"].unique())
        lo, hi = _cluster_bootstrap_group_mean(rows, coco_ids, "cnn_cam_delta_iou", rng)
        result[group_name]["cluster_bootstrap_coco_id_ci_95"] = [lo, hi]
        LOGGER.info("[%s] cluster-bootstrap-on-coco_id CI: [%.4f, %.4f]", group_name, lo, hi)

    interpretation = (
        "DISSOCIATION CONFIRMED - misclassified group's DeltaIoU significantly > 0"
        if result["misclassified"]["wilcoxon_vs_zero_p_value"] is not None and result["misclassified"]["wilcoxon_vs_zero_p_value"] < 0.05
        and result["misclassified"]["mean_delta_iou"] > 0
        else "NOT CONFIRMED - localization and classification appear to fail together in this group"
    )
    result["interpretation"] = interpretation
    LOGGER.info("Fix 1 interpretation: %s", interpretation)

    pd.DataFrame(group_deltas_rows).to_csv(OUTPUT_DIR / "fix1_group_deltas.csv", index=False)
    return result


def _hodges_lehmann(x: np.ndarray, y: np.ndarray) -> float:
    """Vectorized (not a Python double-loop) - needed since this runs inside a 10,000-resample
    bootstrap at n=684; the naive O(n^2) list-comprehension version would take far too long.
    """
    diffs = x - y
    idx_i, idx_j = np.triu_indices(len(diffs))
    pairwise_avgs = (diffs[idx_i] + diffs[idx_j]) / 2.0
    return float(np.median(pairwise_avgs))


def _hodges_lehmann_bootstrap_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(x)
    estimates = []
    for _ in range(N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        estimates.append(_hodges_lehmann(x[idx], y[idx]))
    estimates = np.array(estimates)
    lo, hi = np.percentile(estimates, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    return float(lo), float(hi)


def fix2_random_mask_power(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    LOGGER.info("=== Fix 2: Random-mask stratum - effect size and power ===")
    result: dict = {}

    for mask_type in ("random", "segm", "bbox"):
        subset = df[df["mask_type"] == mask_type]
        paired = subset.groupby("coco_id").agg(cnn=("cnn_cam_delta_iou", "mean"), vit=("vit_rollout_delta_iou", "mean"))
        n = len(paired)
        n_raw_rows = len(subset)
        cnn_vals, vit_vals = paired["cnn"].values, paired["vit"].values
        hl_estimate = _hodges_lehmann(cnn_vals, vit_vals)
        ci_low, ci_high = _hodges_lehmann_bootstrap_ci(cnn_vals, vit_vals, rng)
        diffs = cnn_vals - vit_vals
        sd_diff = float(np.std(diffs, ddof=1))
        avg_rows_per_coco_id = n_raw_rows / n if n else float("nan")

        entry = {
            "n_coco_ids": int(n), "n_raw_rows": int(n_raw_rows), "avg_raw_rows_per_coco_id": avg_rows_per_coco_id,
            "hodges_lehmann_estimate": hl_estimate, "hodges_lehmann_ci_95_bootstrap": [ci_low, ci_high],
            "sd_paired_diff": sd_diff,
        }
        LOGGER.info("[%s] n_coco_ids=%d avg_rows/coco_id=%.2f HL=%.4f CI=[%.4f, %.4f] SD_diff=%.4f",
                    mask_type, n, avg_rows_per_coco_id, hl_estimate, ci_low, ci_high, sd_diff)
        result[mask_type] = entry

    # Minimum detectable effect at 80% power, two-sided alpha=0.05, normal approximation
    # (paired t-test formula; Wilcoxon ARE ~0.955 for near-normal data - reported as an
    # approximation, not an exact Wilcoxon power figure, per spec)
    z_alpha_2 = 1.959964
    z_beta = 0.841621  # 80% power
    n_random = result["random"]["n_coco_ids"]
    sd_random = result["random"]["sd_paired_diff"]
    mde_random = (z_alpha_2 + z_beta) * sd_random / np.sqrt(n_random)
    result["random"]["minimum_detectable_effect_80pct_power"] = float(mde_random)
    LOGGER.info("Random-mask MDE at 80%% power (normal approx): %.4f", mde_random)

    segm_hl, bbox_hl = result["segm"]["hodges_lehmann_estimate"], result["bbox"]["hodges_lehmann_estimate"]
    random_ci_low, random_ci_high = result["random"]["hodges_lehmann_ci_95_bootstrap"]
    comparable_effects_in_ci = (random_ci_low <= segm_hl <= random_ci_high) or (random_ci_low <= bbox_hl <= random_ci_high) or (mde_random >= min(segm_hl, bbox_hl))

    verdict = (
        "UNDERPOWERED - random-mask CI is wide enough to include segm/bbox-comparable effect sizes; "
        "this is not a demonstrated absence of effect"
        if comparable_effects_in_ci else
        "GENUINE NULL - random-mask CI is narrow and excludes segm/bbox-comparable effect sizes; "
        "the object-bias concern plausibly applies to random-geometry regions"
    )
    result["verdict"] = verdict
    LOGGER.info("Fix 2 verdict: %s", verdict)
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df, _floors = _load_data()

    fix1_result = fix1_dissociation_check(df, rng)
    (OUTPUT_DIR / "fix1_dissociation_check.json").write_text(json.dumps(fix1_result, indent=2))
    LOGGER.info("Wrote %s", OUTPUT_DIR / "fix1_dissociation_check.json")

    fix2_result = fix2_random_mask_power(df, rng)
    (OUTPUT_DIR / "fix2_random_mask_power.json").write_text(json.dumps(fix2_result, indent=2))
    LOGGER.info("Wrote %s", OUTPUT_DIR / "fix2_random_mask_power.json")

    (OUTPUT_DIR / "reproducibility.json").write_text(json.dumps({"seed": SEED, "n_resamples": N_RESAMPLES, "ci_level": CI_LEVEL}, indent=2))
    LOGGER.info("Additional analyses complete. Outputs written to %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
