"""Applies FINAL_PROJECT_METHODOLOGY.md Section 14's pre-specified statistical treatment to the
TGIF2 zero-shot localization results: floor-corrected ΔIoU as the primary metric, coco_id as the
unit of analysis, Wilcoxon signed-rank as the single uncorrected primary test, Holm-Bonferroni
correction within each of two secondary-tier families, and cluster-bootstrap (BCa) 95% CIs.

Per TGIF2_STATISTICAL_TREATMENT_SPEC.md Part B. Pure post-hoc analysis over already-computed CSVs -
no model loading, no heatmap generation, no dependency on torch. Primary hypothesis
(FINAL_PROJECT_METHODOLOGY.md Section 13): cnn_cam (Grad-CAM++, CNN branch) localizes diffusion-
inpainting evidence more accurately than vit_rollout (Attention Rollout, ViT branch), on identical
fixed model weights, net of object-shape bias.

Usage:
    python tgif2_statistical_treatment.py
    python tgif2_statistical_treatment.py --dry-run

Requires:
    pip install scipy numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import bootstrap, wilcoxon

from config import Tgif2StatisticalTreatmentConfig, load_tgif2_statistical_treatment_config
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("tgif2_statistical_treatment")

PRIMARY_METHODS = ("cnn_cam", "vit_rollout")


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


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Standard Holm (1979) step-down procedure. No third-party dependency (statsmodels is not
    installed in this environment) - this is a ~10-line algorithm, not worth adding one for.
    Returns adjusted p-values in the same order as the input list.
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


class Tgif2StatisticalTreatment:
    def __init__(self, config: Tgif2StatisticalTreatmentConfig):
        self._config = config

    def run(self, dry_run: bool = False) -> int:
        config = self._config
        set_seed(config.seed)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        per_image_rows = _read_csv(config.per_image_localization_path)
        floor_payload = json.loads(config.floor_values_path.read_text())
        floors = floor_payload["floor"]

        if dry_run:
            per_image_rows = per_image_rows[:100]

        self._floor_correct(per_image_rows, floors)

        primary_result = self._pooled_test(per_image_rows, config)
        primary_result.update(tier="primary", description="cnn_cam vs vit_rollout floor-corrected DeltaIoU, pooled, coco_id-paired")

        mask_type_results = self._stratified_tests(
            per_image_rows, config, group_key=lambda r: r["mask_type"],
        )
        generator_results = self._stratified_tests(
            per_image_rows, config, group_key=lambda r: f"{r['source_share']}/{r['mode']}/{r['gen_model']}",
        )

        exploratory_results = {"splice_vs_regenerated_cnn_cam": self._exploratory_splice_vs_regenerated(per_image_rows, config)}

        results = {
            "n_input_rows": len(per_image_rows),
            "primary": primary_result,
            "secondary": {"mask_type": mask_type_results, "generator": generator_results},
            "exploratory": exploratory_results,
        }

        self._write_results(results)
        self._write_coco_id_deltas(per_image_rows)

        if not dry_run:
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)

        LOGGER.info("TGIF2 statistical treatment complete. Outputs written to %s", config.output_dir)
        return 0

    @staticmethod
    def _floor_correct(rows: list[dict], floors: dict) -> None:
        for row in rows:
            mask_type = row["mask_type"]
            for method in PRIMARY_METHODS:
                iou_key = f"{method}_iou"
                if row.get(iou_key) in (None, ""):
                    continue
                floor_stats = floors.get(method, {}).get(mask_type)
                floor_value = floor_stats["median_iou"] if floor_stats else None
                if floor_value is None:
                    continue
                row[f"{method}_delta_iou"] = float(row[iou_key]) - floor_value

    @staticmethod
    def _collapse_to_coco_id(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            coco_id = row["coco_id"]
            for method in PRIMARY_METHODS:
                key = f"{method}_delta_iou"
                if row.get(key) is not None:
                    grouped[coco_id][method].append(row[key])
        return grouped

    @staticmethod
    def _paired_means(grouped: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
        coco_ids, cnn_vals, vit_vals = [], [], []
        for coco_id, by_method in grouped.items():
            if PRIMARY_METHODS[0] in by_method and PRIMARY_METHODS[1] in by_method:
                coco_ids.append(coco_id)
                cnn_vals.append(float(np.mean(by_method[PRIMARY_METHODS[0]])))
                vit_vals.append(float(np.mean(by_method[PRIMARY_METHODS[1]])))
        return coco_ids, np.array(cnn_vals), np.array(vit_vals)

    def _run_paired_test(self, a_vals: np.ndarray, b_vals: np.ndarray, config: Tgif2StatisticalTreatmentConfig) -> dict:
        n = len(a_vals)
        if n < 2:
            return {"n": n, "error": "insufficient paired data (n < 2)"}

        diffs = a_vals - b_vals
        if np.allclose(diffs, 0.0):
            stat, p_value = 0.0, 1.0
        else:
            stat, p_value = wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="two-sided")

        ci_low: Optional[float] = None
        ci_high: Optional[float] = None
        try:
            bootstrap_result = bootstrap(
                (diffs,), np.mean, n_resamples=config.bootstrap_resamples,
                confidence_level=config.ci_level, method="BCa", random_state=config.seed,
            )
            ci_low = float(bootstrap_result.confidence_interval.low)
            ci_high = float(bootstrap_result.confidence_interval.high)
        except Exception as exc:  # noqa: BLE001 - BCa can fail on degenerate (near-zero-variance) samples
            LOGGER.warning("Bootstrap CI failed for n=%d: %s", n, exc)

        return {
            "n": n,
            "wilcoxon_statistic": float(stat),
            "p_value": float(p_value),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "ci_95_bca": [ci_low, ci_high],
            "significant_at_0.05": bool(p_value < config.alpha),
        }

    def _pooled_test(self, rows: list[dict], config: Tgif2StatisticalTreatmentConfig) -> dict:
        grouped = self._collapse_to_coco_id(rows)
        _coco_ids, cnn_vals, vit_vals = self._paired_means(grouped)
        return self._run_paired_test(cnn_vals, vit_vals, config)

    def _stratified_tests(self, rows: list[dict], config: Tgif2StatisticalTreatmentConfig, group_key) -> dict:
        groups = sorted({group_key(r) for r in rows})
        results: dict[str, dict] = {}
        p_values: list[float] = []
        for group in groups:
            subset = [r for r in rows if group_key(r) == group]
            grouped = self._collapse_to_coco_id(subset)
            _coco_ids, cnn_vals, vit_vals = self._paired_means(grouped)
            result = self._run_paired_test(cnn_vals, vit_vals, config)
            results[group] = result
            p_values.append(result.get("p_value", 1.0))

        adjusted = holm_bonferroni(p_values)
        for (group, result), adj_p in zip(results.items(), adjusted):
            if "error" in result:
                continue
            result["p_value_holm_adjusted"] = adj_p
            result["significant_at_0.05_holm"] = bool(adj_p < config.alpha)
        return results

    def _exploratory_splice_vs_regenerated(self, rows: list[dict], config: Tgif2StatisticalTreatmentConfig) -> dict:
        """cnn_cam floor-corrected DeltaIoU, fully-regenerated (mode=fr) vs spliced (mode=sp),
        flux share only (the only share with both modes), paired on coco_id. Exploratory,
        uncorrected, per Section 14's tier table.
        """
        flux_rows = [r for r in rows if r["source_share"] == "flux"]
        by_coco_mode: dict[str, dict[str, float]] = defaultdict(dict)
        for row in flux_rows:
            value = row.get("cnn_cam_delta_iou")
            if value is not None:
                by_coco_mode[row["coco_id"]][row["mode"]] = value

        fr_vals, sp_vals = [], []
        for by_mode in by_coco_mode.values():
            if "fr" in by_mode and "sp" in by_mode:
                fr_vals.append(by_mode["fr"])
                sp_vals.append(by_mode["sp"])

        result = self._run_paired_test(np.array(fr_vals), np.array(sp_vals), config)
        result.update(tier="exploratory", correction="none", description="cnn_cam DeltaIoU, fully-regenerated (fr) vs spliced (sp), flux share, coco_id-paired")
        return result

    def _write_results(self, results: dict) -> None:
        path = self._config.output_dir / self._config.results_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        LOGGER.info("Wrote statistical results to %s", path)
        primary = results["primary"]
        if "error" not in primary:
            LOGGER.info(
                "[primary] n=%d wilcoxon_stat=%.2f p=%.6g median_diff=%.4f ci95=%s",
                primary["n"], primary["wilcoxon_statistic"], primary["p_value"],
                primary["median_diff"], primary["ci_95_bca"],
            )

    def _write_coco_id_deltas(self, rows: list[dict]) -> None:
        grouped = self._collapse_to_coco_id(rows)
        out_rows = []
        for coco_id, by_method in sorted(grouped.items()):
            row = {"coco_id": coco_id, "n_rows": len(next(iter(by_method.values()), []))}
            for method in PRIMARY_METHODS:
                values = by_method.get(method, [])
                row[f"{method}_delta_iou_mean"] = float(np.mean(values)) if values else None
            out_rows.append(row)
        if not out_rows:
            return
        fieldnames = sorted({key for row in out_rows for key in row.keys()})
        path = self._config.output_dir / self._config.coco_id_level_deltas_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        LOGGER.info("Wrote coco_id-level deltas for %d coco_ids to %s", len(out_rows), path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-image-localization-path", type=str, default=None)
    parser.add_argument("--floor-values-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_tgif2_statistical_treatment_config(
        {
            "per_image_localization_path": args.per_image_localization_path,
            "floor_values_path": args.floor_values_path,
            "output_dir": args.output_dir,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / "tgif2_statistical_treatment.log")
    LOGGER.info("Loaded configuration")

    if not config.per_image_localization_path.exists():
        LOGGER.error("Per-image localization CSV not found: %s (run tgif2_localization_eval.py first)", config.per_image_localization_path)
        return 1
    if not config.floor_values_path.exists():
        LOGGER.error("Floor values JSON not found: %s (run tgif2_object_bias_floor.py first)", config.floor_values_path)
        return 1

    try:
        return Tgif2StatisticalTreatment(config).run(dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during TGIF2 statistical treatment:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
