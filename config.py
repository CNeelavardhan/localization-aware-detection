"""Configuration for the binary (real vs. manipulated) classifier training phases.

EfficientNetB3Config is frozen per EFFICIENTNET_B3_SPEC.md, ViTB16Config per VIT_B16_SPEC.md.
Paths and hardware-dependent values (batch size, workers) may be overridden via CLI flags in
the train_*.py / evaluate.py scripts; architecture and optimization hyperparameters are fixed
for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class EfficientNetB3Config:
    # Paths
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("outputs/efficientnet_b3")

    # Input
    image_size: int = 224
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Augmentation (train split only)
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 10.0
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    random_resized_crop_scale_min: float = 0.8

    # Model
    dropout: float = 0.3
    dense_units: int = 256

    # Training
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 50
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    early_stopping_patience: int = 7
    early_stopping_metric: str = "f1"
    num_workers: int = 4
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    best_model_filename: str = "best_model.pth"
    training_log_filename: str = "training.log"
    epoch_history_filename: str = "epoch_history.csv"
    curves_filename: str = "loss_accuracy_curves.png"
    predictions_filename: str = "predictions.csv"
    features_filename: str = "feature_vectors.npz"
    metrics_filename: str = "metrics.json"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "EfficientNetB3Config":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("data_dir", "splits_dir", "output_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)

    @property
    def split_csv_paths(self) -> dict[str, Path]:
        return {name: self.splits_dir / f"{name}.csv" for name in ("train", "val", "test")}


def load_config(overrides: Optional[dict[str, Any]] = None) -> EfficientNetB3Config:
    return EfficientNetB3Config().with_overrides(**(overrides or {}))


load_efficientnet_b3_config = load_config


@dataclass(frozen=True)
class ViTB16Config:
    # Paths
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("outputs/vit_b16")

    # Input
    image_size: int = 224
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Augmentation (train split only)
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 10.0
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    random_resized_crop_scale_min: float = 0.8

    # Model
    dropout: float = 0.3
    dense_units: int = 256

    # Training
    backbone_lr: float = 5e-5
    head_lr: float = 1e-3
    batch_size: int = 8
    max_epochs: int = 50
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    early_stopping_patience: int = 7
    early_stopping_metric: str = "f1"
    num_workers: int = 0
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    best_model_filename: str = "best_model.pth"
    training_log_filename: str = "training.log"
    epoch_history_filename: str = "epoch_history.csv"
    curves_filename: str = "loss_accuracy_curves.png"
    predictions_filename: str = "predictions.csv"
    features_filename: str = "feature_vectors.npz"
    metrics_filename: str = "metrics.json"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "ViTB16Config":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("data_dir", "splits_dir", "output_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)

    @property
    def split_csv_paths(self) -> dict[str, Path]:
        return {name: self.splits_dir / f"{name}.csv" for name in ("train", "val", "test")}


def load_vit_b16_config(overrides: Optional[dict[str, Any]] = None) -> ViTB16Config:
    return ViTB16Config().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class HybridFusionConfig:
    # Paths
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("outputs/hybrid_fusion")
    efficientnet_b3_checkpoint: Path = Path("outputs/efficientnet_b3/best_model.pth")
    vit_b16_checkpoint: Path = Path("outputs/vit_b16/best_model.pth")

    # Input
    image_size: int = 224
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Augmentation (train split only)
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 10.0
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    random_resized_crop_scale_min: float = 0.8

    # Backbones (must match EfficientNetB3Config/ViTB16Config's trained architecture)
    backbone_dropout: float = 0.3
    backbone_dense_units: int = 256
    cnn_feature_dim: int = 1536
    vit_feature_dim: int = 768

    # Fusion
    projection_dim: int = 256
    fusion_hidden_dim: int = 256
    dropout: float = 0.3

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 8
    max_epochs: int = 50
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    early_stopping_patience: int = 7
    early_stopping_metric: str = "f1"
    num_workers: int = 0
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    best_model_filename: str = "best_model.pth"
    training_log_filename: str = "training.log"
    epoch_history_filename: str = "epoch_history.csv"
    curves_filename: str = "loss_accuracy_curves.png"
    predictions_filename: str = "predictions.csv"
    features_filename: str = "feature_vectors.npz"
    metrics_filename: str = "metrics.json"
    reproducibility_filename: str = "reproducibility.json"
    comparison_filename: str = "comparison_metrics.json"

    def with_overrides(self, **overrides: Any) -> "HybridFusionConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("data_dir", "splits_dir", "output_dir", "efficientnet_b3_checkpoint", "vit_b16_checkpoint"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)

    @property
    def split_csv_paths(self) -> dict[str, Path]:
        return {name: self.splits_dir / f"{name}.csv" for name in ("train", "val", "test")}


def load_hybrid_fusion_config(overrides: Optional[dict[str, Any]] = None) -> HybridFusionConfig:
    return HybridFusionConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class GradCAMConfig:
    # Paths
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("outputs/gradcam")
    hybrid_fusion_checkpoint: Path = Path("outputs/hybrid_fusion/best_model.pth")

    # Visualization
    overlay_alpha: float = 0.45

    # Runtime
    num_workers: int = 0
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    summary_filename: str = "gradcam_summary.csv"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "GradCAMConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("data_dir", "splits_dir", "output_dir", "hybrid_fusion_checkpoint"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_gradcam_config(overrides: Optional[dict[str, Any]] = None) -> GradCAMConfig:
    return GradCAMConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class LocalizationEvalConfig:
    # Paths
    data_dir: Path = Path("data")
    gradcam_dir: Path = Path("outputs/gradcam")
    output_dir: Path = Path("outputs/localization_eval")

    # Thresholding
    binarization_threshold: float = 0.5
    threshold_sweep: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    # Qualitative outputs
    num_qualitative_examples: int = 5

    # Runtime
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    metrics_filename: str = "localization_metrics.json"
    per_image_filename: str = "per_image_localization.csv"
    threshold_sweep_filename: str = "threshold_sweep.csv"
    qualitative_dirname: str = "qualitative_examples"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "LocalizationEvalConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("data_dir", "gradcam_dir", "output_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_localization_eval_config(overrides: Optional[dict[str, Any]] = None) -> LocalizationEvalConfig:
    return LocalizationEvalConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class AttentionRolloutConfig:
    # Paths
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("outputs/attention_rollout")
    hybrid_fusion_checkpoint: Path = Path("outputs/hybrid_fusion/best_model.pth")

    # Visualization
    overlay_alpha: float = 0.45

    # Runtime
    num_workers: int = 0
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    summary_filename: str = "attention_rollout_summary.csv"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "AttentionRolloutConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("data_dir", "splits_dir", "output_dir", "hybrid_fusion_checkpoint"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_attention_rollout_config(overrides: Optional[dict[str, Any]] = None) -> AttentionRolloutConfig:
    return AttentionRolloutConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class ObjectBiasFloorConfig:
    # Paths
    data_dir: Path = Path("data")
    gradcam_dir: Path = Path("outputs/gradcam")
    attention_rollout_dir: Path = Path("outputs/attention_rollout")
    localization_eval_dir: Path = Path("outputs/localization_eval")
    output_dir: Path = Path("outputs/object_bias_floor")

    # Thresholding (matches LocalizationEvalConfig's convention)
    binarization_threshold: float = 0.5

    # Runtime
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    floor_values_filename: str = "floor_values.json"
    real_image_scores_filename: str = "real_image_scores.csv"
    delta_iou_filename: str = "delta_iou.csv"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "ObjectBiasFloorConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("data_dir", "gradcam_dir", "attention_rollout_dir", "localization_eval_dir", "output_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_object_bias_floor_config(overrides: Optional[dict[str, Any]] = None) -> ObjectBiasFloorConfig:
    return ObjectBiasFloorConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class Tgif2LocalizationEvalConfig:
    # Paths
    manifest_path: Path = Path("manifests/tgif2_manifest.csv")
    tgif_dataset_dir: Path = Path("tgif-dataset")
    output_dir: Path = Path("outputs/tgif2_localization_eval")
    hybrid_fusion_checkpoint: Path = Path("outputs/hybrid_fusion/best_model.pth")
    # data_dir/splits_dir are unused for TGIF2 (no v1 data involved) but load_hybrid_fusion_config
    # requires them to construct the model architecture; point at v1's own paths for that purpose
    # only - no v1 image is ever read by this phase.
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")

    # Scope (TGIF2_LOCALIZATION_EVAL_SPEC.md: var_id=0 only, both splits)
    var_id_filter: str = "0"

    # Thresholding (matches LocalizationEvalConfig's convention)
    binarization_threshold: float = 0.5

    # Visualization
    overlay_alpha: float = 0.45
    save_overlays: bool = False  # raw .npz only by default; 4,788 x 3 overlay PNGs is a lot of I/O

    # Runtime
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    per_image_filename: str = "tgif2_per_image_localization.csv"
    metrics_filename: str = "tgif2_localization_metrics.json"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "Tgif2LocalizationEvalConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("manifest_path", "tgif_dataset_dir", "output_dir", "hybrid_fusion_checkpoint", "data_dir", "splits_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_tgif2_localization_eval_config(overrides: Optional[dict[str, Any]] = None) -> Tgif2LocalizationEvalConfig:
    return Tgif2LocalizationEvalConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class Tgif2ObjectBiasFloorConfig:
    # Paths
    manifest_path: Path = Path("manifests/tgif2_manifest.csv")
    tgif_dataset_dir: Path = Path("tgif-dataset")
    output_dir: Path = Path("outputs/tgif2_object_bias_floor")
    hybrid_fusion_checkpoint: Path = Path("outputs/hybrid_fusion/best_model.pth")
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")

    # Scope (matches Tgif2LocalizationEvalConfig's convention)
    var_id_filter: str = "0"

    # Thresholding (matches LocalizationEvalConfig's convention)
    binarization_threshold: float = 0.5

    # Runtime
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    floor_values_filename: str = "tgif2_floor_values.json"
    authentic_image_scores_filename: str = "tgif2_authentic_image_scores.csv"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "Tgif2ObjectBiasFloorConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("manifest_path", "tgif_dataset_dir", "output_dir", "hybrid_fusion_checkpoint", "data_dir", "splits_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_tgif2_object_bias_floor_config(overrides: Optional[dict[str, Any]] = None) -> Tgif2ObjectBiasFloorConfig:
    return Tgif2ObjectBiasFloorConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class Tgif2StatisticalTreatmentConfig:
    # Paths
    per_image_localization_path: Path = Path("outputs/tgif2_localization_eval/tgif2_per_image_localization.csv")
    floor_values_path: Path = Path("outputs/tgif2_object_bias_floor/tgif2_floor_values.json")
    output_dir: Path = Path("outputs/tgif2_statistical_treatment")

    # Statistics
    bootstrap_resamples: int = 10000
    ci_level: float = 0.95
    alpha: float = 0.05

    # Runtime
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    results_filename: str = "tgif2_statistical_results.json"
    coco_id_level_deltas_filename: str = "tgif2_coco_id_level_deltas.csv"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "Tgif2StatisticalTreatmentConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("per_image_localization_path", "floor_values_path", "output_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_tgif2_statistical_treatment_config(overrides: Optional[dict[str, Any]] = None) -> Tgif2StatisticalTreatmentConfig:
    return Tgif2StatisticalTreatmentConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class TraceFeasibilityTestConfig:
    # Paths
    manifest_path: Path = Path("manifests/tgif2_manifest.csv")
    tgif_dataset_dir: Path = Path("tgif-dataset")
    output_dir: Path = Path("outputs/trace_feasibility_test")

    # Sampling (TRACE_FEASIBILITY_TEST_SPEC.md Section 10)
    sample_size_per_group: int = 50
    num_positive_controls: int = 10
    mask_type: str = "segm"
    seed: int = 42

    # Residual method: "bilateral" (primary, user's confirmed choice) or "srm" (fallback, per
    # TRACE_FEASIBILITY_TEST_SPEC.md - the Ker/KV 5x5 high-pass predictor kernel from steganalysis)
    residual_method: str = "bilateral"
    bilateral_sigma_color: float = 0.05
    bilateral_sigma_spatial: float = 5.0

    # Region statistics / sliding-window heatmap
    dilation_radius_px: int = 32
    window_size_px: int = 32
    heatmap_size: int = 224
    binarization_threshold: float = 0.5

    # JPEG robustness check
    jpeg_quality: int = 90

    # Runtime
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    scores_filename: str = "trace_scores.csv"
    jpeg_robustness_filename: str = "jpeg_robustness.csv"
    decision_rule_filename: str = "decision_rule_result.json"
    inspection_dir_name: str = "inspection_heatmaps"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "TraceFeasibilityTestConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("manifest_path", "tgif_dataset_dir", "output_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_trace_feasibility_test_config(overrides: Optional[dict[str, Any]] = None) -> TraceFeasibilityTestConfig:
    return TraceFeasibilityTestConfig().with_overrides(**(overrides or {}))


@dataclass(frozen=True)
class AntiforensicsConfig:
    # Paths - Part A (in-distribution)
    gradcam_summary_path: Path = Path("outputs/gradcam/gradcam_summary.csv")
    metadata_dir: Path = Path("data/metadata")
    manipulated_dir: Path = Path("data/manipulated")
    hybrid_fusion_checkpoint: Path = Path("outputs/hybrid_fusion/best_model.pth")
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")

    # Paths - Part B (TGIF2 zero-shot)
    manifest_path: Path = Path("manifests/tgif2_manifest.csv")
    tgif_dataset_dir: Path = Path("tgif-dataset")
    tgif2_floor_values_path: Path = Path("outputs/tgif2_object_bias_floor/tgif2_floor_values.json")

    output_dir: Path = Path("outputs/antiforensics")

    # Part B sampling (3 strata with segm masks - "random" share has no segm/bbox, only "random"
    # mask_type, discovered during implementation; ANTIFORENSICS_SPEC.md's 4-strata assumption
    # corrected to 3)
    part_b_strata: tuple = (("tgif", "fr", "sdxl"), ("flux", "fr", "flux1filldev"), ("flux", "sp", "flux1filldev"))
    part_b_sample_per_stratum: int = 50
    part_b_mask_type: str = "segm"
    var_id_filter: str = "0"

    # Thresholding (matches established convention)
    binarization_threshold: float = 0.5

    # Pre-registered thresholds (ANTIFORENSICS_SPEC.md Sections 3-4)
    detection_evasion_recall_threshold: float = 0.5
    localization_suppression_floor_threshold: float = 0.0

    # Statistics
    bootstrap_resamples: int = 10000
    ci_level: float = 0.95
    alpha: float = 0.05

    # Runtime
    seed: int = 42
    log_level: str = "INFO"

    # Output filenames (written under output_dir)
    detection_by_attack_filename: str = "detection_by_attack.csv"
    localization_by_attack_filename: str = "localization_by_attack.csv"
    heatmap_displacement_filename: str = "heatmap_displacement.csv"
    summary_filename: str = "summary.md"
    examples_dir_name: str = "examples"
    reproducibility_filename: str = "reproducibility.json"

    def with_overrides(self, **overrides: Any) -> "AntiforensicsConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for path_field in ("gradcam_summary_path", "metadata_dir", "manipulated_dir", "hybrid_fusion_checkpoint",
                            "data_dir", "splits_dir", "manifest_path", "tgif_dataset_dir",
                            "tgif2_floor_values_path", "output_dir"):
            if path_field in clean:
                clean[path_field] = Path(clean[path_field])
        return replace(self, **clean)


def load_antiforensics_config(overrides: Optional[dict[str, Any]] = None) -> AntiforensicsConfig:
    return AntiforensicsConfig().with_overrides(**(overrides or {}))
