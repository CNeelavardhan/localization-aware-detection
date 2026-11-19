"""Anti-forensics robustness evaluation, per ANTIFORENSICS_SPEC.md (which itself implements
ANTIFORENSICS_BRIEF.md, with one logged, user-confirmed deviation - see the spec's Section 0).

Two independent parts, treating standard post-processing as deliberate adversarial attack rather
than incidental degradation, per the brief's forensic threat model (ANTIFORENSICS_SPEC.md Section 1):

Part A - Detection evasion: does the classifier still label an attacked image as manipulated? Runs
on the 176-image in-distribution test set (not TGIF2 - TGIF2's zero-shot clean-baseline recall is
already 7.8%, discovered this session, too close to floor for a meaningful degradation curve; see
outputs/tgif2_classification_recall_gap in memory and the spec's Section 0).

Part B - Localization evidence suppression: does the heatmap still point at the true region, and
does it move, under attack? Runs on TGIF2 zero-shot (~150 images, 3 strata), exactly as the brief
specified - this question is independent of classification accuracy (Grad-CAM++ backpropagates the
raw logit regardless of the thresholded prediction).

Both parts are exploratory tier, reported uncorrected, not competing with the Section 4.9.2 primary
hypothesis. gradcam.py, attention_rollout.py, localization_eval.py, tgif2_localization_eval.py, and
experiments.py stay frozen - only imported.

Usage:
    python antiforensics_eval.py --dry-run
    python antiforensics_eval.py --part-a-only
    python antiforensics_eval.py --part-b-only
    python antiforensics_eval.py

Requires:
    pip install torch torchvision scikit-learn numpy scipy pillow
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon

from attention_rollout import AttentionRolloutGenerator
from config import AntiforensicsConfig, load_antiforensics_config, load_hybrid_fusion_config
from dataset import TransformFactory
from experiments import SignificanceTester
from gradcam import GradCAMPlusPlus, HeatmapVisualizer
from localization_eval import HEATMAP_TYPES as GRADCAM_HEATMAP_TYPES
from localization_eval import LocalizationMetricsCalculator, MaskLoader
from tgif2_localization_eval import Tgif2ManifestLoader
from train_hybrid_fusion import HybridFusionClassifier
from training_utils import ReproducibilityRecorder, set_seed

LOGGER = logging.getLogger("antiforensics_eval")

# Attack families and levels, ANTIFORENSICS_SPEC.md Section 2. Ordered within each family by
# increasing severity, so "first level crossing a threshold" is well-defined.
ATTACKS: list[tuple[str, str, dict]] = [
    ("jpeg", "q95", {"quality": 95}), ("jpeg", "q85", {"quality": 85}), ("jpeg", "q75", {"quality": 75}),
    ("jpeg", "q65", {"quality": 65}), ("jpeg", "q50", {"quality": 50}), ("jpeg", "q35", {"quality": 35}),
    ("downscale", "0.9x", {"factor": 0.9}), ("downscale", "0.75x", {"factor": 0.75}), ("downscale", "0.5x", {"factor": 0.5}),
    ("blur", "sigma0.5", {"sigma": 0.5}), ("blur", "sigma1.0", {"sigma": 1.0}), ("blur", "sigma1.5", {"sigma": 1.5}),
    ("noise", "sigma2", {"sigma": 2.0}), ("noise", "sigma5", {"sigma": 5.0}), ("noise", "sigma10", {"sigma": 10.0}),
]


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


class AttackGenerator:
    """Standard consumer-grade image processing, applied as deliberate attack per the threat model
    (ANTIFORENSICS_SPEC.md Section 1) rather than incidental degradation. Pure PIL/numpy/scipy - no
    new dependencies.
    """

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def apply(self, image: Image.Image, family: str, params: dict) -> Image.Image:
        if family == "jpeg":
            return self._jpeg_recompress(image, params["quality"])
        if family == "downscale":
            return self._downscale_upscale(image, params["factor"])
        if family == "blur":
            return self._gaussian_blur(image, params["sigma"])
        if family == "noise":
            return self._additive_noise(image, params["sigma"])
        raise ValueError(f"Unknown attack family: {family}")

    @staticmethod
    def _jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def _downscale_upscale(image: Image.Image, factor: float) -> Image.Image:
        w, h = image.size
        small_w, small_h = max(1, round(w * factor)), max(1, round(h * factor))
        small = image.convert("RGB").resize((small_w, small_h), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    @staticmethod
    def _gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
        blurred = gaussian_filter(arr, sigma=(sigma, sigma, 0))
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))

    def _additive_noise(self, image: Image.Image, sigma: float) -> Image.Image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
        noise = self._rng.normal(0.0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


class DetectionEvasionRunner:
    """Part A: in-distribution test set, 176 manipulated images. Measures whether attacks flip the
    classifier's decision from manipulated to authentic.
    """

    def __init__(self, config: AntiforensicsConfig, attack_generator: AttackGenerator, device: torch.device):
        self._config = config
        self._attacks = attack_generator
        self._device = device

    def run(self, model: HybridFusionClassifier, transform, dry_run: bool) -> tuple[list[dict], dict]:
        config = self._config
        rows = [r for r in _read_csv(config.gradcam_summary_path) if r["y_true"] == "1"]
        LOGGER.info("Part A: %d in-distribution manipulated test images loaded", len(rows))
        if dry_run:
            rows = rows[:10]

        results: list[dict] = []
        for idx, row in enumerate(rows):
            image_id, file_name = row["image_id"], row["file_name"]
            manipulation_type = self._manipulation_type(image_id)
            image_path = config.manipulated_dir / file_name
            with Image.open(image_path) as img:
                clean_image = img.convert("RGB")

            clean_prob = float(row["y_prob"])
            results.append({
                "image_id": image_id, "manipulation_type": manipulation_type,
                "attack_family": "clean", "attack_level": "clean", "y_prob": clean_prob,
                "y_pred": int(clean_prob >= config.binarization_threshold),
            })

            for family, level, params in ATTACKS:
                attacked_image = self._attacks.apply(clean_image, family, params)
                image_tensor = transform(attacked_image).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    logit = model(image_tensor).squeeze()
                    y_prob = float(torch.sigmoid(logit).item())
                results.append({
                    "image_id": image_id, "manipulation_type": manipulation_type,
                    "attack_family": family, "attack_level": level, "y_prob": y_prob,
                    "y_pred": int(y_prob >= config.binarization_threshold),
                })

            if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
                LOGGER.info("Part A: processed %d/%d images", idx + 1, len(rows))

        thresholds = self._compute_thresholds(results)
        return results, thresholds

    def _manipulation_type(self, image_id: str) -> str:
        meta_path = self._config.metadata_dir / f"{int(image_id):012d}.json"
        if not meta_path.exists():
            return "unknown"
        return json.loads(meta_path.read_text()).get("manipulation_type", "unknown")

    def _compute_thresholds(self, rows: list[dict]) -> dict:
        by_condition: dict[tuple, list[dict]] = {}
        for row in rows:
            by_condition.setdefault((row["attack_family"], row["attack_level"]), []).append(row)

        clean_by_image = {r["image_id"]: r["y_pred"] for r in rows if r["attack_family"] == "clean"}
        image_order = sorted(clean_by_image)
        y_true = np.ones(len(image_order), dtype=int)
        clean_pred = np.array([clean_by_image[i] for i in image_order])

        thresholds: dict = {}
        for family in sorted({f for f, _l, _p in ATTACKS}):
            family_levels = [(f, l, p) for f, l, p in ATTACKS if f == family]
            crossing_level = None
            per_level = []
            for f, level, _params in family_levels:
                condition_rows = by_condition[(f, level)]
                attacked_by_image = {r["image_id"]: r["y_pred"] for r in condition_rows}
                attacked_pred = np.array([attacked_by_image[i] for i in image_order])
                recall = float(np.mean(attacked_pred))
                mcnemar = SignificanceTester.mcnemar_test(y_true, clean_pred, attacked_pred)
                per_level.append({"level": level, "recall": recall, "mcnemar_p_value": mcnemar["p_value"]})
                if crossing_level is None and recall < self._config.detection_evasion_recall_threshold:
                    crossing_level = level
            thresholds[family] = {"per_level": per_level, "first_level_below_threshold": crossing_level}

        clean_recall = float(np.mean(clean_pred))
        thresholds["clean_baseline_recall"] = clean_recall
        return thresholds


class LocalizationSuppressionRunner:
    """Part B: TGIF2 zero-shot, 3 strata (segm masks only - the "random" share has no segm/bbox
    masks, discovered during implementation; ANTIFORENSICS_SPEC.md's 4-strata assumption corrected
    to 3, logged here rather than silently).
    """

    def __init__(self, config: AntiforensicsConfig, attack_generator: AttackGenerator, device: torch.device):
        self._config = config
        self._attacks = attack_generator
        self._device = device

    def run(self, model: HybridFusionClassifier, transform, cam_generator: GradCAMPlusPlus,
            rollout_generator: AttentionRolloutGenerator, manifest_loader: "Tgif2ManifestLoader",
            dry_run: bool) -> tuple[list[dict], list[dict]]:
        config = self._config
        LOGGER.info("Part B strata (corrected from spec's 4 to 3 - 'random' share has no segm masks): %s", config.part_b_strata)

        floor_payload = json.loads(config.tgif2_floor_values_path.read_text())
        floor_segm = {method: floor_payload["floor"][method]["segm"]["median_iou"] for method in ("cnn_cam", "vit_rollout")}

        rows = manifest_loader.load_rows(config.manifest_path, config.var_id_filter)
        by_stratum: dict[tuple, dict] = {}
        for row in rows:
            key = (row["source_share"], row["mode"], row["gen_model"])
            if key in config.part_b_strata and row["mask_type"] == config.part_b_mask_type:
                by_stratum.setdefault(key, {})[row["coco_id"]] = row

        rng = random.Random(config.seed)
        sample_size = 3 if dry_run else config.part_b_sample_per_stratum
        sampled_rows: list[dict] = []
        for stratum, coco_map in by_stratum.items():
            coco_ids = sorted(coco_map)
            rng.shuffle(coco_ids)
            for coco_id in coco_ids[:sample_size]:
                sampled_rows.append(coco_map[coco_id])
        LOGGER.info("Part B: sampled %d images across %d strata", len(sampled_rows), len(by_stratum))

        localization_rows: list[dict] = []
        displacement_rows: list[dict] = []

        for idx, row in enumerate(sampled_rows):
            resolved = manifest_loader.resolve(row)
            if resolved is None:
                continue
            _orig_path, mask_path = resolved
            mask_224 = MaskLoader.load_binary_mask(mask_path, size=224)
            stratum_label = f"{row['source_share']}/{row['mode']}/{row['gen_model']}"

            with Image.open(Path(row["file_path"])) as img:
                clean_image = img.convert("RGB")

            clean_heatmaps = self._generate_heatmaps(clean_image, transform, cam_generator, rollout_generator)
            clean_binarized = {m: (clean_heatmaps[m] >= config.binarization_threshold) for m in ("cnn_cam", "vit_rollout")}

            for method in ("cnn_cam", "vit_rollout"):
                clean_metrics = LocalizationMetricsCalculator.compute(mask_224, clean_heatmaps[method], config.binarization_threshold)
                if clean_metrics is not None:
                    localization_rows.append({
                        "coco_id": row["coco_id"], "stratum": stratum_label, "method": method,
                        "attack_family": "clean", "attack_level": "clean", "iou": clean_metrics.iou,
                        "delta_iou": clean_metrics.iou - floor_segm[method], "auc": clean_metrics.auc,
                    })

            for family, level, params in ATTACKS:
                attacked_image = self._attacks.apply(clean_image, family, params)
                attacked_heatmaps = self._generate_heatmaps(attacked_image, transform, cam_generator, rollout_generator)

                for method in ("cnn_cam", "vit_rollout"):
                    metrics = LocalizationMetricsCalculator.compute(mask_224, attacked_heatmaps[method], config.binarization_threshold)
                    if metrics is not None:
                        delta_iou = metrics.iou - floor_segm[method]
                        localization_rows.append({
                            "coco_id": row["coco_id"], "stratum": stratum_label, "method": method,
                            "attack_family": family, "attack_level": level, "iou": metrics.iou,
                            "delta_iou": delta_iou, "auc": metrics.auc,
                        })

                    displacement_metrics = LocalizationMetricsCalculator.compute(
                        clean_binarized[method], attacked_heatmaps[method], config.binarization_threshold,
                    )
                    displacement_rows.append({
                        "coco_id": row["coco_id"], "stratum": stratum_label, "method": method,
                        "attack_family": family, "attack_level": level,
                        "displacement_iou": displacement_metrics.iou if displacement_metrics is not None else None,
                    })

            if (idx + 1) % 20 == 0 or (idx + 1) == len(sampled_rows):
                LOGGER.info("Part B: processed %d/%d images", idx + 1, len(sampled_rows))

        return localization_rows, displacement_rows

    def _generate_heatmaps(self, image: Image.Image, transform, cam_generator: GradCAMPlusPlus,
                            rollout_generator: AttentionRolloutGenerator) -> dict:
        image_tensor = transform(image).unsqueeze(0)
        cam_result = cam_generator.generate(image_tensor)
        vit_rollout = rollout_generator.generate(image_tensor)
        return {"cnn_cam": cam_result["cnn_cam"], "vit_rollout": vit_rollout}


class AntiforensicsRunner:
    def __init__(self, config: AntiforensicsConfig):
        self._config = config

    def run(self, dry_run: bool, part_a_only: bool, part_b_only: bool) -> int:
        config = self._config
        set_seed(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Using device: %s", device)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / config.examples_dir_name).mkdir(parents=True, exist_ok=True)

        fusion_config = load_hybrid_fusion_config({"data_dir": str(config.data_dir), "splits_dir": str(config.splits_dir)})
        model = HybridFusionClassifier(fusion_config)
        model.load_state_dict(torch.load(config.hybrid_fusion_checkpoint, map_location=device))
        model.to(device)
        model.eval()
        transform = TransformFactory.build(fusion_config, train=False)
        attack_generator = AttackGenerator(config.seed)

        detection_rows: list[dict] = []
        detection_thresholds: dict = {}
        localization_rows: list[dict] = []
        displacement_rows: list[dict] = []

        if not part_b_only:
            LOGGER.info("=== Part A: Detection Evasion (in-distribution) ===")
            part_a = DetectionEvasionRunner(config, attack_generator, device)
            detection_rows, detection_thresholds = part_a.run(model, transform, dry_run)
            self._write_detection_csv(detection_rows)
            self._save_example_attacks(detection_rows, config)

        if not part_a_only:
            LOGGER.info("=== Part B: Localization Evidence Suppression (TGIF2 zero-shot) ===")
            cam_generator = GradCAMPlusPlus(model, device)
            rollout_generator = AttentionRolloutGenerator(model, device)
            manifest_loader = Tgif2ManifestLoader(config.tgif_dataset_dir)
            part_b = LocalizationSuppressionRunner(config, attack_generator, device)
            localization_rows, displacement_rows = part_b.run(model, transform, cam_generator, rollout_generator, manifest_loader, dry_run)
            self._write_localization_csv(localization_rows)
            self._write_displacement_csv(displacement_rows)

        localization_thresholds = self._compute_localization_thresholds(localization_rows) if localization_rows else {}
        self._write_summary(detection_thresholds, localization_thresholds, part_a_only, part_b_only)

        if not dry_run:
            ReproducibilityRecorder.record(config, config.output_dir / config.reproducibility_filename)

        LOGGER.info("Anti-forensics evaluation complete. Outputs written to %s", config.output_dir)
        return 0

    def _compute_localization_thresholds(self, rows: list[dict]) -> dict:
        # Paired by coco_id, cnn_cam only (the primary hypothesis's method), per
        # ANTIFORENSICS_SPEC.md Section 5 - Wilcoxon signed-rank, clean vs attacked, per level.
        cnn_rows = [r for r in rows if r["method"] == "cnn_cam"]
        clean_by_coco = {r["coco_id"]: r["delta_iou"] for r in cnn_rows if r["attack_family"] == "clean"}

        by_condition: dict[tuple, dict[str, float]] = {}
        for row in cnn_rows:
            if row["attack_family"] == "clean":
                continue
            by_condition.setdefault((row["attack_family"], row["attack_level"]), {})[row["coco_id"]] = row["delta_iou"]

        thresholds: dict = {}
        for family in sorted({f for f, _l, _p in ATTACKS}):
            family_levels = [(f, l, p) for f, l, p in ATTACKS if f == family]
            crossing_level = None
            per_level = []
            for f, level, _params in family_levels:
                attacked_by_coco = by_condition.get((f, level), {})
                paired_ids = sorted(set(clean_by_coco) & set(attacked_by_coco))
                clean_vals = np.array([clean_by_coco[c] for c in paired_ids])
                attacked_vals = np.array([attacked_by_coco[c] for c in paired_ids])
                median_delta = float(np.median(attacked_vals)) if len(attacked_vals) else None

                p_value = None
                if len(paired_ids) >= 2 and not np.allclose(clean_vals, attacked_vals):
                    _stat, p_value = wilcoxon(clean_vals, attacked_vals, zero_method="wilcox", alternative="two-sided")
                    p_value = float(p_value)

                per_level.append({"level": level, "median_delta_iou": median_delta, "n": len(paired_ids), "wilcoxon_p_value": p_value})
                if crossing_level is None and median_delta is not None and median_delta < self._config.localization_suppression_floor_threshold:
                    crossing_level = level
            thresholds[family] = {"per_level": per_level, "first_level_below_floor": crossing_level}
        return thresholds

    def _save_example_attacks(self, detection_rows: list[dict], config: AntiforensicsConfig) -> None:
        example_image_id = detection_rows[0]["image_id"] if detection_rows else None
        if example_image_id is None:
            return
        summary_rows = _read_csv(config.gradcam_summary_path)
        file_name = next((r["file_name"] for r in summary_rows if r["image_id"] == example_image_id), None)
        if file_name is None:
            return
        image_path = config.manipulated_dir / file_name
        if not image_path.exists():
            return
        attack_gen = AttackGenerator(config.seed)
        with Image.open(image_path) as img:
            clean = img.convert("RGB")
        examples_dir = config.output_dir / config.examples_dir_name
        clean.save(examples_dir / f"{example_image_id}_clean.png")
        for family, level, params in [("jpeg", "q35", {"quality": 35}), ("blur", "sigma1.5", {"sigma": 1.5}), ("noise", "sigma10", {"sigma": 10.0})]:
            attacked = attack_gen.apply(clean, family, params)
            attacked.save(examples_dir / f"{example_image_id}_{family}_{level}.png")
        LOGGER.info("Saved example attacked images for visual inspection to %s", examples_dir)

    def _write_detection_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        path = self._config.output_dir / self._config.detection_by_attack_filename
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %d detection-evasion rows to %s", len(rows), path)

    def _write_localization_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        path = self._config.output_dir / self._config.localization_by_attack_filename
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %d localization-suppression rows to %s", len(rows), path)

    def _write_displacement_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        path = self._config.output_dir / self._config.heatmap_displacement_filename
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %d heatmap-displacement rows to %s", len(rows), path)

    def _write_summary(self, detection_thresholds: dict, localization_thresholds: dict, part_a_only: bool, part_b_only: bool) -> None:
        lines = ["# Anti-Forensics Evaluation Summary\n"]
        lines.append("Exploratory tier, per ANTIFORENSICS_SPEC.md - not competing with the Section 4.9.2 primary hypothesis.\n")
        if detection_thresholds and not part_b_only:
            lines.append("## Part A: Detection Evasion (in-distribution test set)\n")
            lines.append(f"Clean baseline recall: {detection_thresholds.get('clean_baseline_recall', 'n/a'):.4f}\n")
            lines.append(f"Pre-registered threshold: recall < {self._config.detection_evasion_recall_threshold}\n")
            for family in sorted({f for f, _l, _p in ATTACKS}):
                data = detection_thresholds.get(family, {})
                lines.append(f"- **{family}**: first level below threshold = {data.get('first_level_below_threshold', 'none reached')}")
        if localization_thresholds and not part_a_only:
            lines.append("\n## Part B: Localization Evidence Suppression (TGIF2 zero-shot, cnn_cam)\n")
            lines.append(f"Pre-registered threshold: median DeltaIoU < {self._config.localization_suppression_floor_threshold} (crosses below object-bias floor)\n")
            for family in sorted({f for f, _l, _p in ATTACKS}):
                data = localization_thresholds.get(family, {})
                lines.append(f"- **{family}**: first level below floor = {data.get('first_level_below_floor', 'none reached')}")
        if not part_a_only and not part_b_only:
            lines.append("\n## Does Localization Fail Before Detection?\n")
            lines.append("Structural comparison (different image populations, per ANTIFORENSICS_SPEC.md Section 4) - "
                          "compare the attack-level thresholds above directly; see detection_by_attack.csv and "
                          "localization_by_attack.csv for the full per-level curves this summary condenses.\n")
        path = self._config.output_dir / self._config.summary_filename
        path.write_text("\n".join(lines))
        LOGGER.info("Wrote summary to %s", path)
        for line in lines:
            LOGGER.info(line)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--part-a-only", action="store_true")
    parser.add_argument("--part-b-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_antiforensics_config({"output_dir": args.output_dir, "log_level": args.log_level})
    configure_logging(config.log_level, config.output_dir / "antiforensics_eval.log")
    LOGGER.info("Loaded configuration")

    if not config.hybrid_fusion_checkpoint.exists():
        LOGGER.error("Hybrid fusion checkpoint not found: %s", config.hybrid_fusion_checkpoint)
        return 1

    try:
        return AntiforensicsRunner(config).run(dry_run=args.dry_run, part_a_only=args.part_a_only, part_b_only=args.part_b_only)
    except Exception:  # noqa: BLE001 - top-level guard so the traceback is captured in the log file
        LOGGER.error("Fatal error during anti-forensics evaluation:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
