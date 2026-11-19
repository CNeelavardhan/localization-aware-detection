"""External baseline: CNNDetection (Wang, Wang, Zhang, Owens & Efros, CVPR 2020), a published
pretrained ResNet50 GAN-image detector, evaluated zero-shot (no fine-tuning) on this project's own
test split. Addresses THESIS_DISCUSSION.md's "No external baseline" limitation (Section 6.4) — a
same-protocol comparison against a real published detector, not just this project's own three
models against each other.

Pretrained weights: blur_jpg_prob0.5.pth (the paper's most robust variant, trained with blur/JPEG
augmentation), downloaded from the authors' released checkpoints
(https://github.com/PeterWang512/CNNDetection). The checkpoint loads directly into
torchvision's own resnet50(num_classes=1) - no third-party network code needed.

Preprocessing follows the authors' own demo.py exactly: ToTensor + ImageNet normalization, no
resize or crop (their default), since the model is a fully-convolutional ResNet50 with global
average pooling and accepts any input resolution. This means this baseline sees images at native
COCO resolution, unlike this project's own three models (which resize to 224x224) - an
intentional divergence, since re-implementing the baseline's own preprocessing choices as-published
is what makes this a faithful reproduction of their reported behavior, not a modified variant.

Usage:
    python baseline_cnndetection.py
    python baseline_cnndetection.py --weights external_weights/cnndetection_blur_jpg_prob0.5.pth

Requires:
    pip install torch torchvision scikit-learn
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from config import load_config
from dataset import RealFakeDataset
from evaluate import ModelEvaluator, ResultWriter

LOGGER = logging.getLogger("baseline_cnndetection")


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    LOGGER.addHandler(handler)


class CNNDetectionBaseline(torch.nn.Module):
    """Wraps torchvision's resnet50(num_classes=1) to match this project's other models'
    forward(x, return_features=True) -> (logits, features) convention, so evaluate.py's
    ModelEvaluator can be reused unchanged.
    """

    def __init__(self, weights_path: Path):
        super().__init__()
        self.backbone = resnet50(weights=None, num_classes=1)
        state_dict = torch.load(weights_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict["model"])

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone.avgpool(
            self.backbone.layer4(
                self.backbone.layer3(
                    self.backbone.layer2(
                        self.backbone.layer1(
                            self.backbone.maxpool(
                                self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
                            )
                        )
                    )
                )
            )
        )
        feats = torch.flatten(features, 1)
        logits = self.backbone.fc(feats)
        if return_features:
            return logits, feats
        return logits


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--weights", type=str, default="external_weights/cnndetection_blur_jpg_prob0.5.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline_cnndetection")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        LOGGER.error("Baseline weights not found: %s", weights_path)
        return 1

    config = load_config({"data_dir": args.data_dir, "splits_dir": args.splits_dir})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    LOGGER.info("Loading CNNDetection (Wang et al., CVPR 2020) pretrained ResNet50, zero-shot (no fine-tuning)")
    model = CNNDetectionBaseline(weights_path)
    model.to(device)
    model.eval()

    # No resize/crop - the authors' own default preprocessing, native COCO resolution
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    dataset = RealFakeDataset(config.split_csv_paths[args.split], config.data_dir, transform)
    # batch_size=1: images are at native (variable) resolution, cannot be collated into a batch
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    LOGGER.info("Evaluating on '%s' split: %d samples (native resolution, batch_size=1)", args.split, len(dataset))

    result = ModelEvaluator(model, device).evaluate(loader)

    LOGGER.info(
        "[BASELINE: CNNDetection] split=%s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        args.split, result.metrics.accuracy, result.metrics.precision,
        result.metrics.recall, result.metrics.f1, result.metrics.roc_auc,
    )
    LOGGER.info("[BASELINE: CNNDetection] confusion_matrix=%s", result.metrics.confusion_matrix)

    ResultWriter.write_predictions(output_dir / f"{args.split}_predictions.csv", result, args.split)
    ResultWriter.write_metrics(output_dir / f"{args.split}_metrics.json", result.metrics)
    LOGGER.info("Wrote predictions and metrics to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
