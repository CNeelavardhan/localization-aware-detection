"""Evaluation utilities and standalone test-set evaluation for the EfficientNet-B3 classifier.

Usage:
    python evaluate.py --config-override '{"data_dir": "data"}' \
        --checkpoint outputs/efficientnet_b3/best_model.pth --split test

Requires (not needed to import ClassificationMetrics/MetricsCalculator alone):
    pip install torch torchvision scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import EfficientNetB3Config, load_config

LOGGER = logging.getLogger("evaluate")


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    LOGGER.addHandler(handler)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsCalculator:
    """Computes standard binary classification metrics from labels/probabilities."""

    @staticmethod
    def compute(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = (y_prob >= threshold).astype(int)
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else float("nan")

        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=roc_auc,
            confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        )


@dataclass(frozen=True)
class EvaluationResult:
    image_ids: list[int]
    y_true: np.ndarray
    y_prob: np.ndarray
    features: np.ndarray
    metrics: ClassificationMetrics
    loss: Optional[float] = None


class ModelEvaluator:
    """Runs a trained model over a DataLoader, collecting predictions, features, and metrics.

    Pass `criterion` to also compute the average loss in the same forward pass, instead of
    requiring a second pass over the data just to get a loss value.
    """

    def __init__(self, model: Any, device: Any):
        self._model = model
        self._device = device

    def evaluate(self, data_loader: Any, criterion: Optional[Any] = None) -> EvaluationResult:
        import torch

        self._model.eval()
        image_ids: list[int] = []
        y_true: list[int] = []
        y_prob: list[float] = []
        features: list[np.ndarray] = []
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for images, labels, ids in data_loader:
                images = images.to(self._device)
                logits, feats = self._model(images, return_features=True)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

                if criterion is not None:
                    batch_labels = labels.to(self._device).float().unsqueeze(1)
                    loss = criterion(logits, batch_labels)
                    total_loss += loss.item() * images.size(0)
                    total_count += images.size(0)

                y_prob.extend(probs.tolist())
                y_true.extend(labels.tolist())
                image_ids.extend(ids.tolist())
                features.append(feats.cpu().numpy())

        y_true_arr = np.array(y_true, dtype=int)
        y_prob_arr = np.array(y_prob, dtype=float)
        features_arr = np.concatenate(features, axis=0) if features else np.empty((0,))
        metrics = MetricsCalculator.compute(y_true_arr, y_prob_arr)
        avg_loss = (total_loss / total_count) if criterion is not None and total_count else None

        return EvaluationResult(
            image_ids=image_ids, y_true=y_true_arr, y_prob=y_prob_arr, features=features_arr,
            metrics=metrics, loss=avg_loss,
        )


class ResultWriter:
    @staticmethod
    def write_predictions(path: Path, result: EvaluationResult, split: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "split", "y_true", "y_prob", "y_pred"])
            for image_id, y_true, y_prob in zip(result.image_ids, result.y_true, result.y_prob):
                writer.writerow([image_id, split, int(y_true), f"{y_prob:.6f}", int(y_prob >= 0.5)])

    @staticmethod
    def write_metrics(path: Path, metrics: ClassificationMetrics) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics.to_dict(), indent=2))

    @staticmethod
    def write_features(path: Path, result: EvaluationResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            image_ids=np.array(result.image_ids, dtype=int),
            y_true=result.y_true,
            features=result.features,
        )


def _build_model_and_loader(config: EfficientNetB3Config, checkpoint_path: Path, split: str):
    import torch
    from torch.utils.data import DataLoader

    from dataset import build_dataset
    from train_efficientnet_b3 import EfficientNetB3Classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB3Classifier(dropout=config.dropout, dense_units=config.dense_units)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    dataset = build_dataset(config, split)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return model, loader, device


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <output_dir>/best_model.pth")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--log-level", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(
        {
            "data_dir": args.data_dir,
            "splits_dir": args.splits_dir,
            "output_dir": args.output_dir,
            "log_level": args.log_level,
        }
    )
    configure_logging(config.log_level)

    checkpoint_path = args.checkpoint or (config.output_dir / config.best_model_filename)
    if not checkpoint_path.exists():
        LOGGER.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    LOGGER.info("Evaluating checkpoint %s on split '%s'", checkpoint_path, args.split)
    model, loader, device = _build_model_and_loader(config, checkpoint_path, args.split)
    result = ModelEvaluator(model, device).evaluate(loader)

    LOGGER.info(
        "Split '%s' | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        args.split,
        result.metrics.accuracy,
        result.metrics.precision,
        result.metrics.recall,
        result.metrics.f1,
        result.metrics.roc_auc,
    )
    LOGGER.info("Confusion matrix: %s", result.metrics.confusion_matrix)

    ResultWriter.write_predictions(config.output_dir / f"{args.split}_{config.predictions_filename}", result, args.split)
    ResultWriter.write_metrics(config.output_dir / f"{args.split}_{config.metrics_filename}", result.metrics)
    ResultWriter.write_features(config.output_dir / f"{args.split}_{config.features_filename}", result)

    LOGGER.info("Wrote predictions, metrics, and feature vectors to %s", config.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
