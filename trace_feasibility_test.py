"""The trace feasibility test (E0) — the gating step from FINAL_PROJECT_METHODOLOGY.md Part III
that decides how much of the forensic trace layer (Section 7) and correlation layer (Section 8)
get built. Classical noise-residual analysis, completely independent of the trained model (no
torch, no HybridFusionClassifier) so any later agreement between this and the model's heatmaps is
corroborating evidence, not circular reasoning.

Per TRACE_FEASIBILITY_TEST_SPEC.md: 50 spliced (flux/sp) + 50 fully-regenerated (tgif/fr) + 50
authentic images, paired by coco_id, all scored against segm masks. Residual = bilateral-filter
denoiser residual (user's confirmed choice). Region statistics (inside mask vs. dilated band) give
`delta`; a sliding-window heatmap (uniform_filter approximation) gives AUC/IoU/Dice/Precision/Recall
via localization_eval.py's existing scoring machinery, unchanged. Plus: 10 positive-control images
(crude copy-paste splices, script-generated) the pipeline must detect, a JPEG Q=90 robustness
check on the spliced group, and 10 by-eye inspection visualizations.

The frozen decision rule (Section 11) is applied to the actual results at the end, not decided in
advance.

Usage:
    python trace_feasibility_test.py
    python trace_feasibility_test.py --dry-run
    python trace_feasibility_test.py --inspect-only

Requires:
    pip install numpy scipy scikit-image pillow scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, convolve, uniform_filter
from scipy.stats import mannwhitneyu
from skimage.restoration import denoise_bilateral
from skimage.transform import resize as sk_resize

# The Ker/KV 5x5 high-pass predictor kernel (Fridrich & Kodovsky, 2012; "SRM-style high-pass" per
# TRACE_FEASIBILITY_TEST_SPEC.md's documented fallback) - the canonical residual filter from
# steganalysis literature, also used as a first-layer constraint in forensic CNNs (Bayar & Stamm).
SRM_KV_KERNEL = np.array([
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1],
], dtype=np.float64) / 12.0

from config import TraceFeasibilityTestConfig, load_trace_feasibility_test_config
from gradcam import HeatmapVisualizer
from localization_eval import LocalizationMetricsCalculator, MaskLoader
from tgif2_localization_eval import Tgif2ManifestLoader
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("trace_feasibility_test")


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


def _load_grayscale_float(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float64) / 255.0


def _load_native_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        mask_img = img.convert("L")
        if mask_img.size != (shape[1], shape[0]):
            mask_img = mask_img.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(mask_img) >= 128


class ResidualPipeline:
    """The core, model-independent forensic signal: bilateral-filter residual -> region
    statistics -> sliding-window localization heatmap. Same logic applied to every image
    regardless of group (spliced/regenerated/authentic/positive_control) - the pipeline must not
    know which group an image belongs to.
    """

    def __init__(self, config: TraceFeasibilityTestConfig):
        self._config = config

    def _compute_residual(self, grayscale: np.ndarray) -> np.ndarray:
        if self._config.residual_method == "srm":
            filtered = convolve(grayscale, SRM_KV_KERNEL, mode="reflect")
            return np.abs(filtered)
        denoised = denoise_bilateral(
            grayscale, sigma_color=self._config.bilateral_sigma_color,
            sigma_spatial=self._config.bilateral_sigma_spatial, channel_axis=None,
        )
        return np.abs(grayscale - denoised)

    def score(self, image_path: Path, mask_path: Path, save_inspection_to: Optional[Path] = None) -> Optional[dict]:
        grayscale = _load_grayscale_float(image_path)
        residual = self._compute_residual(grayscale)
        mask_native = _load_native_mask(mask_path, residual.shape)

        delta = self._region_delta(residual, mask_native)
        heatmap_224 = self._sliding_window_heatmap(residual)
        mask_224 = MaskLoader.load_binary_mask(mask_path, size=self._config.heatmap_size)
        metrics = LocalizationMetricsCalculator.compute(mask_224, heatmap_224, self._config.binarization_threshold)

        if save_inspection_to is not None:
            with Image.open(image_path) as original:
                HeatmapVisualizer.save_overlay(original.convert("RGB"), heatmap_224, save_inspection_to, alpha=0.45)

        result = {"delta": delta, "mask_area_px": int(mask_native.sum())}
        for field, value in (metrics.to_dict().items() if metrics is not None else []):
            result[f"trace_{field}"] = value
        return result

    def score_array(self, grayscale: np.ndarray, mask_native: np.ndarray, rgb_image: Optional[np.ndarray] = None,
                     save_inspection_to: Optional[Path] = None) -> dict:
        """Same as score(), but takes an already-loaded grayscale array + native mask directly -
        used for positive controls, which are synthesized in-memory rather than read from disk.
        """
        residual = self._compute_residual(grayscale)
        delta = self._region_delta(residual, mask_native)
        heatmap_224 = self._sliding_window_heatmap(residual)
        mask_224 = self._resize_mask_nearest(mask_native, self._config.heatmap_size)
        metrics = LocalizationMetricsCalculator.compute(mask_224, heatmap_224, self._config.binarization_threshold)

        if save_inspection_to is not None and rgb_image is not None:
            HeatmapVisualizer.save_overlay(Image.fromarray(rgb_image), heatmap_224, save_inspection_to, alpha=0.45)

        result = {"delta": delta, "mask_area_px": int(mask_native.sum())}
        for field, value in (metrics.to_dict().items() if metrics is not None else []):
            result[f"trace_{field}"] = value
        return result

    def _region_delta(self, residual: np.ndarray, mask: np.ndarray) -> Optional[float]:
        if not mask.any():
            return None
        dilated = binary_dilation(mask, iterations=self._config.dilation_radius_px)
        band = dilated & ~mask
        inside_mean = float(residual[mask].mean())
        if not band.any():
            return None
        band_mean = float(residual[band].mean())
        return inside_mean - band_mean

    def _sliding_window_heatmap(self, residual: np.ndarray) -> np.ndarray:
        local_mean = uniform_filter(residual, size=self._config.window_size_px)
        lo, hi = float(local_mean.min()), float(local_mean.max())
        normalized = (local_mean - lo) / (hi - lo) if hi > lo else np.zeros_like(local_mean)
        resized = sk_resize(normalized, (self._config.heatmap_size, self._config.heatmap_size), anti_aliasing=True)
        return np.clip(resized, 0.0, 1.0)

    @staticmethod
    def _resize_mask_nearest(mask: np.ndarray, size: int) -> np.ndarray:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize((size, size), Image.NEAREST)
        return np.asarray(mask_img) >= 128


class PositiveControlGenerator:
    """Crude, non-AI copy-paste splices: a rectangular patch copied from elsewhere in the same
    authentic image and pasted at a new non-overlapping location, no blending. Deliberately crude
    - the point is "does the pipeline detect an obvious splice at all."
    """

    def __init__(self, rng: random.Random, patch_area_px: float):
        self._rng = rng
        side = max(8, int(round(patch_area_px ** 0.5)))
        self._patch_w = side
        self._patch_h = side

    def generate(self, target_rgb: np.ndarray, source_rgb: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Pastes a patch into target_rgb. If source_rgb is given (a DIFFERENT photo, resized to
        match target_rgb's shape by the caller), the patch is copied from there - a genuine
        cross-source splice, which is what a noise-residual method is actually designed to catch
        (same-image self-copy has no cross-sensor noise mismatch to find; see
        trace_feasibility_test_attempt.md in memory for why this matters). Falls back to
        same-image copy if source_rgb is None.
        """
        h, w, _ = target_rgb.shape
        source = source_rgb if source_rgb is not None else target_rgb
        pw, ph = min(self._patch_w, w // 3), min(self._patch_h, h // 3)

        for _ in range(200):
            src_x = self._rng.randint(0, w - pw)
            src_y = self._rng.randint(0, h - ph)
            dst_x = self._rng.randint(0, w - pw)
            dst_y = self._rng.randint(0, h - ph)
            if source_rgb is not None:
                break  # different image entirely - no self-overlap concern
            overlaps = not (dst_x + pw <= src_x or src_x + pw <= dst_x or dst_y + ph <= src_y or src_y + ph <= dst_y)
            if not overlaps:
                break

        spliced = target_rgb.copy()
        patch = source[src_y:src_y + ph, src_x:src_x + pw].copy()
        spliced[dst_y:dst_y + ph, dst_x:dst_x + pw] = patch

        mask = np.zeros((h, w), dtype=bool)
        mask[dst_y:dst_y + ph, dst_x:dst_x + pw] = True
        return spliced, mask


class TraceFeasibilityRunner:
    def __init__(self, config: TraceFeasibilityTestConfig):
        self._config = config
        self._pipeline = ResidualPipeline(config)

    def run(self, dry_run: bool = False, inspect_only: bool = False, positive_control_only: bool = False) -> int:
        config = self._config
        set_seed(config.seed)
        rng = random.Random(config.seed)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        manifest_loader = Tgif2ManifestLoader(config.tgif_dataset_dir)
        rows = manifest_loader.load_rows(config.manifest_path, "0")

        flux_sp = {r["coco_id"]: r for r in rows if r["source_share"] == "flux" and r["mode"] == "sp" and r["mask_type"] == config.mask_type}
        tgif_fr = {r["coco_id"]: r for r in rows if r["source_share"] == "tgif" and r["mode"] == "fr" and r["mask_type"] == config.mask_type}
        eligible = sorted(set(flux_sp) & set(tgif_fr))
        LOGGER.info("%d coco_ids eligible (present in both flux/sp and tgif/fr at mask_type=%s)", len(eligible), config.mask_type)

        sample_size = 3 if dry_run else config.sample_size_per_group
        num_positive = 2 if dry_run else config.num_positive_controls

        rng.shuffle(eligible)
        sample_ids = eligible[:sample_size]
        remaining = eligible[sample_size:]
        positive_control_ids = remaining[:num_positive]
        authentic_comparison_ids = remaining[num_positive:2 * num_positive]  # for --positive-control-only

        inspection_dir = config.output_dir / config.inspection_dir_name
        inspection_dir.mkdir(parents=True, exist_ok=True)
        num_to_inspect = 10 if not dry_run else 3
        inspected_so_far = 0

        if positive_control_only:
            return self._run_positive_control_validation(
                positive_control_ids, authentic_comparison_ids, tgif_fr, manifest_loader, rng, inspection_dir,
            )

        rows_out: list[dict] = []
        mask_areas: list[int] = []

        for coco_id in sample_ids:
            for group, row in (("spliced", flux_sp[coco_id]), ("regenerated", tgif_fr[coco_id])):
                resolved = manifest_loader.resolve(row)
                if resolved is None:
                    LOGGER.warning("Could not resolve orig/mask for coco_id=%s group=%s, skipping", coco_id, group)
                    continue
                _orig_path, mask_path = resolved
                image_path = Path(row["file_path"])
                save_to = None
                if inspected_so_far < num_to_inspect:
                    save_to = inspection_dir / f"{group}_{coco_id}.png"
                    inspected_so_far += 1
                scored = self._pipeline.score(image_path, mask_path, save_inspection_to=save_to)
                if scored is not None:
                    rows_out.append({"coco_id": coco_id, "group": group, **scored})
                    if scored.get("mask_area_px"):
                        mask_areas.append(scored["mask_area_px"])

            # Authentic: the tgif orig image, scored against the SAME segm mask as "regenerated"
            resolved = manifest_loader.resolve(tgif_fr[coco_id])
            if resolved is None:
                continue
            orig_path, mask_path = resolved
            save_to = None
            if inspected_so_far < num_to_inspect:
                save_to = inspection_dir / f"authentic_{coco_id}.png"
                inspected_so_far += 1
            scored = self._pipeline.score(orig_path, mask_path, save_inspection_to=save_to)
            if scored is not None:
                rows_out.append({"coco_id": coco_id, "group": "authentic", **scored})

        LOGGER.info("Scored %d/%d groups across %d coco_ids", len(rows_out), 3 * len(sample_ids), len(sample_ids))

        median_mask_area = float(np.median(mask_areas)) if mask_areas else 4096.0
        LOGGER.info("Median segm mask area across main sample: %.1f px^2", median_mask_area)

        if inspect_only:
            self._write_scores(rows_out)
            LOGGER.info("--inspect-only: stopping after inspection heatmaps + main sample scoring. Look at %s before running the full pipeline.", inspection_dir)
            return 0

        # Positive controls - cross-image splices (patch from a DIFFERENT photo, not the same
        # image), per trace_feasibility_test_attempt.md's finding that same-image copy has no
        # genuine cross-source noise mismatch for a residual method to detect.
        pc_generator = PositiveControlGenerator(rng, median_mask_area)
        for idx, coco_id in enumerate(positive_control_ids):
            source_coco_id = positive_control_ids[(idx + len(positive_control_ids) // 2) % len(positive_control_ids)]
            spliced_rgb, pc_mask = self._make_cross_image_splice(coco_id, source_coco_id, tgif_fr, manifest_loader, pc_generator)
            if spliced_rgb is None:
                continue
            grayscale = np.asarray(Image.fromarray(spliced_rgb).convert("L"), dtype=np.float64) / 255.0
            save_to = inspection_dir / f"positive_control_{coco_id}.png" if idx < 3 else None
            scored = self._pipeline.score_array(grayscale, pc_mask, rgb_image=spliced_rgb, save_inspection_to=save_to)
            rows_out.append({"coco_id": coco_id, "group": "positive_control", **scored})

        self._write_scores(rows_out)

        jpeg_rows = self._jpeg_robustness_check(sample_ids, flux_sp, manifest_loader, dry_run)
        self._write_jpeg_robustness(jpeg_rows)

        decision = self._apply_decision_rule(rows_out)
        self._write_decision(decision)

        if not dry_run:
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)

        LOGGER.info("Trace feasibility test complete. Outputs written to %s", config.output_dir)
        return 0

    @staticmethod
    def _make_cross_image_splice(target_coco_id: str, source_coco_id: str, tgif_fr: dict,
                                  manifest_loader: Tgif2ManifestLoader, pc_generator: PositiveControlGenerator) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        resolved_target = manifest_loader.resolve(tgif_fr[target_coco_id]) if target_coco_id in tgif_fr else None
        resolved_source = manifest_loader.resolve(tgif_fr[source_coco_id]) if source_coco_id in tgif_fr else None
        if resolved_target is None or resolved_source is None:
            return None, None
        target_orig_path, _ = resolved_target
        source_orig_path, _ = resolved_source
        with Image.open(target_orig_path) as img:
            target_rgb = np.asarray(img.convert("RGB"))
        with Image.open(source_orig_path) as img:
            source_rgb = np.asarray(img.convert("RGB").resize((target_rgb.shape[1], target_rgb.shape[0])))
        return pc_generator.generate(target_rgb, source_rgb=source_rgb)

    def _run_positive_control_validation(self, positive_control_ids: list[str], authentic_ids: list[str], tgif_fr: dict,
                                          manifest_loader: Tgif2ManifestLoader, rng: random.Random, inspection_dir: Path) -> int:
        """Validates the current residual_method against the positive control BEFORE committing to
        the full 150-image run, per user's explicit instruction when approving the SRM fallback:
        confirm it actually detects planted splices first.
        """
        LOGGER.info("--positive-control-only: validating residual_method=%s against %d cross-image splices vs %d authentic images",
                    self._config.residual_method, len(positive_control_ids), len(authentic_ids))

        pc_generator = PositiveControlGenerator(rng, patch_area_px=16000.0)  # matches the full run's actual median segm area
        pc_rows: list[dict] = []
        for idx, coco_id in enumerate(positive_control_ids):
            source_coco_id = positive_control_ids[(idx + len(positive_control_ids) // 2) % len(positive_control_ids)]
            spliced_rgb, pc_mask = self._make_cross_image_splice(coco_id, source_coco_id, tgif_fr, manifest_loader, pc_generator)
            if spliced_rgb is None:
                continue
            grayscale = np.asarray(Image.fromarray(spliced_rgb).convert("L"), dtype=np.float64) / 255.0
            save_to = inspection_dir / f"pc_validation_{self._config.residual_method}_{coco_id}.png" if idx < 3 else None
            scored = self._pipeline.score_array(grayscale, pc_mask, rgb_image=spliced_rgb, save_inspection_to=save_to)
            pc_rows.append({"coco_id": coco_id, "group": "positive_control", **scored})

        auth_rows: list[dict] = []
        for coco_id in authentic_ids:
            resolved = manifest_loader.resolve(tgif_fr[coco_id]) if coco_id in tgif_fr else None
            if resolved is None:
                continue
            orig_path, mask_path = resolved
            scored = self._pipeline.score(orig_path, mask_path)
            if scored is not None:
                auth_rows.append({"coco_id": coco_id, "group": "authentic", **scored})

        pc_delta = [r["delta"] for r in pc_rows if r.get("delta") is not None]
        auth_delta = [r["delta"] for r in auth_rows if r.get("delta") is not None]
        pc_iou = [r["trace_iou"] for r in pc_rows if r.get("trace_iou") is not None]
        auth_iou = [r["trace_iou"] for r in auth_rows if r.get("trace_iou") is not None]

        result: dict = {"residual_method": self._config.residual_method, "n_positive_control": len(pc_delta), "n_authentic": len(auth_delta)}
        if len(pc_delta) >= 2 and len(auth_delta) >= 2:
            stat, p = mannwhitneyu(pc_delta, auth_delta, alternative="two-sided")
            rb = 1.0 - (2.0 * stat) / (len(pc_delta) * len(auth_delta))
            result["delta_mannwhitney"] = {"u_statistic": float(stat), "p_value": float(p), "rank_biserial": float(rb)}
            result["separates"] = bool(p < 0.05 and rb > 0)
        if len(pc_iou) >= 2 and len(auth_iou) >= 2:
            stat, p = mannwhitneyu(pc_iou, auth_iou, alternative="two-sided")
            result["iou_mannwhitney"] = {"u_statistic": float(stat), "p_value": float(p)}
        result["pc_delta_median"] = float(np.median(pc_delta)) if pc_delta else None
        result["auth_delta_median"] = float(np.median(auth_delta)) if auth_delta else None

        path = self._config.output_dir / f"positive_control_validation_{self._config.residual_method}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        LOGGER.info("Positive control validation (%s): separates=%s, %s", self._config.residual_method, result.get("separates"), result.get("delta_mannwhitney"))
        LOGGER.info("Wrote %s", path)
        return 0

    def _jpeg_robustness_check(self, sample_ids: list[str], flux_sp: dict, manifest_loader: Tgif2ManifestLoader, dry_run: bool) -> list[dict]:
        rows_out = []
        ids = sample_ids[:3] if dry_run else sample_ids
        for coco_id in ids:
            row = flux_sp.get(coco_id)
            if row is None:
                continue
            resolved = manifest_loader.resolve(row)
            if resolved is None:
                continue
            _orig_path, mask_path = resolved
            image_path = Path(row["file_path"])

            original_scored = self._pipeline.score(image_path, mask_path)
            if original_scored is None or original_scored.get("delta") is None:
                continue

            recompressed_path = self._config.output_dir / "_jpeg_tmp.jpg"
            with Image.open(image_path) as img:
                img.convert("RGB").save(recompressed_path, "JPEG", quality=self._config.jpeg_quality)
            recompressed_scored = self._pipeline.score(recompressed_path, mask_path)
            recompressed_path.unlink(missing_ok=True)

            rows_out.append({
                "coco_id": coco_id,
                "delta_original": original_scored["delta"],
                "delta_jpeg_q90": recompressed_scored["delta"] if recompressed_scored else None,
            })
        return rows_out

    @staticmethod
    def _apply_decision_rule(rows: list[dict]) -> dict:
        def deltas_for(group: str) -> list[float]:
            return [r["delta"] for r in rows if r["group"] == group and r.get("delta") is not None]

        def ious_for(group: str) -> list[float]:
            return [r["trace_iou"] for r in rows if r["group"] == group and r.get("trace_iou") is not None]

        spliced_delta, regen_delta, auth_delta = deltas_for("spliced"), deltas_for("regenerated"), deltas_for("authentic")
        spliced_iou, regen_iou, auth_iou = ious_for("spliced"), ious_for("regenerated"), ious_for("authentic")

        def mw_test(a: list[float], b: list[float]) -> dict:
            if len(a) < 2 or len(b) < 2:
                return {"error": "insufficient data"}
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            n1, n2 = len(a), len(b)
            rank_biserial = 1.0 - (2.0 * stat) / (n1 * n2)
            return {"u_statistic": float(stat), "p_value": float(p), "rank_biserial": float(rank_biserial), "n1": n1, "n2": n2}

        spliced_vs_auth = mw_test(spliced_delta, auth_delta)
        regen_vs_auth = mw_test(regen_delta, auth_delta)

        floor_delta = float(np.median(auth_delta)) if auth_delta else None
        floor_iou = float(np.median(auth_iou)) if auth_iou else None
        spliced_median_iou = float(np.median(spliced_iou)) if spliced_iou else None

        spliced_separates = bool(spliced_vs_auth.get("p_value", 1.0) < 0.05 and spliced_vs_auth.get("rank_biserial", 0) > 0)
        regen_separates = bool(regen_vs_auth.get("p_value", 1.0) < 0.05 and regen_vs_auth.get("rank_biserial", 0) > 0)
        clears_floor = bool(spliced_median_iou is not None and floor_iou is not None and spliced_median_iou > floor_iou)

        if regen_separates:
            outcome = "D"
            action = "Fully-regenerated separates too - do not believe it, check mask pairing for a bug before proceeding"
        elif spliced_separates and clears_floor:
            outcome = "A"
            action = "Build the full trace and correlation layers as designed"
        elif spliced_separates and not clears_floor:
            outcome = "B"
            action = "Build a reduced layer - trace corroborates presence, not location"
        else:
            outcome = "C"
            action = "Drop the trace-corroboration claim; report as a negative finding (this is legitimate and publishable, not a failed test)"

        return {
            "outcome": outcome, "action": action,
            "spliced_vs_authentic": spliced_vs_auth, "regenerated_vs_authentic": regen_vs_auth,
            "floor_delta": floor_delta, "floor_iou": floor_iou, "spliced_median_iou": spliced_median_iou,
            "spliced_separates": spliced_separates, "regenerated_separates": regen_separates, "clears_floor": clears_floor,
        }

    def _write_scores(self, rows: list[dict]) -> None:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        path = self._config.output_dir / self._config.scores_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %d trace scores to %s", len(rows), path)

    def _write_jpeg_robustness(self, rows: list[dict]) -> None:
        if not rows:
            return
        path = self._config.output_dir / self._config.jpeg_robustness_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote JPEG robustness check for %d images to %s", len(rows), path)

    def _write_decision(self, decision: dict) -> None:
        path = self._config.output_dir / self._config.decision_rule_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision, indent=2))
        LOGGER.info("Decision rule outcome: %s - %s", decision["outcome"], decision["action"])
        LOGGER.info("Wrote decision rule result to %s", path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--tgif-dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--residual-method", type=str, default=None, choices=["bilateral", "srm"])
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--positive-control-only", action="store_true", help="Score just the 10 positive controls + a matched authentic sample, to validate the residual method before the full run")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_trace_feasibility_test_config(
        {
            "manifest_path": args.manifest_path,
            "tgif_dataset_dir": args.tgif_dataset_dir,
            "output_dir": args.output_dir,
            "residual_method": args.residual_method,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level, config.output_dir / "trace_feasibility_test.log")
    LOGGER.info("Loaded configuration (residual_method=%s)", config.residual_method)

    if not config.manifest_path.exists():
        LOGGER.error("TGIF2 manifest not found: %s", config.manifest_path)
        return 1

    try:
        return TraceFeasibilityRunner(config).run(dry_run=args.dry_run, inspect_only=args.inspect_only, positive_control_only=args.positive_control_only)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during trace feasibility test:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
