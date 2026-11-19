"""External baseline #2: UniversalFakeDetect (Ojha, Li & Lee, CVPR 2023) - a frozen CLIP:ViT-L/14
image encoder with a linear probe trained only on GAN-generated images (ProGAN), evaluated
zero-shot (no fine-tuning) on this project's own test split. Chosen over DIRE (Wang et al., ICCV
2023) as the diffusion-specific external baseline because DIRE's own pretrained classifier weights
are only distributed via BaiduDrive/RecDrive (not reliably retrievable outside China without an
account; the 2023 OneDrive mirror is dead), whereas UniversalFakeDetect's weights
(pretrained_weights/fc_weights.pth, vendored into external_weights/) are committed directly to the
authors' public GitHub repository - fully reproducible with no gated hosting. UniversalFakeDetect's
own stated contribution is explicitly that CLIP features trained on GAN images generalize to
detecting *diffusion*-generated images without diffusion images ever appearing in training - a
directly relevant prior claim to test against this thesis's own diffusion-manipulation setting.

Source (MIT License): https://github.com/Yuheng-Li/UniversalFakeDetect
The CLIP implementation (external_models/universalfakedetect/clip/) is vendored unmodified from
that repository for a faithful reproduction; the CLIP:ViT-L/14 backbone weights are downloaded
automatically (from OpenAI's own public host) on first run and cached under ~/.cache/clip.

Preprocessing follows the authors' own validate.py exactly: CenterCrop(224) + CLIP-specific
normalization (not ImageNet's) - not this thesis's own 224x224 resize convention, since faithfully
reproducing the baseline's published preprocessing is what makes this a fair test of the released
detector.

Usage:
    python baseline_universalfakedetect.py
    python baseline_universalfakedetect.py --fc-weights external_weights/universalfakedetect_fc_weights.pth

Requires:
    pip install torch torchvision scikit-learn ftfy regex
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent / "external_models" / "universalfakedetect"))
from clip import clip  # noqa: E402 - vendored CLIP, path set above

from config import load_config
from dataset import RealFakeDataset
from evaluate import ModelEvaluator, ResultWriter

LOGGER = logging.getLogger("baseline_universalfakedetect")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
CLIP_FEATURE_DIM = 768  # ViT-L/14


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    LOGGER.addHandler(handler)


class UniversalFakeDetectBaseline(nn.Module):
    """Frozen CLIP:ViT-L/14 image encoder + linear probe, matching this project's other models'
    forward(x, return_features=True) -> (logits, features) convention.
    """

    def __init__(self, fc_weights_path: Path):
        super().__init__()
        LOGGER.info("Downloading/loading CLIP:ViT-L/14 (cached under ~/.cache/clip after first run)...")
        self.clip_model, _ = clip.load("ViT-L/14", device="cpu")
        self.fc = nn.Linear(CLIP_FEATURE_DIM, 1)
        self.fc.load_state_dict(torch.load(fc_weights_path, map_location="cpu"))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.clip_model.encode_image(x).float()
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--fc-weights", type=str, default="external_weights/universalfakedetect_fc_weights.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline_universalfakedetect")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    fc_weights_path = Path(args.fc_weights)
    if not fc_weights_path.exists():
        LOGGER.error("Linear-probe weights not found: %s", fc_weights_path)
        return 1

    config = load_config({"data_dir": args.data_dir, "splits_dir": args.splits_dir})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    LOGGER.info("Loading UniversalFakeDetect (Ojha et al., CVPR 2023), zero-shot (no fine-tuning)")
    model = UniversalFakeDetectBaseline(fc_weights_path)
    model.to(device)
    model.eval()

    # Authors' own validate.py preprocessing: CenterCrop(224) + CLIP normalization, no resize
    transform = transforms.Compose(
        [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)]
    )
    dataset = RealFakeDataset(config.split_csv_paths[args.split], config.data_dir, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    LOGGER.info("Evaluating on '%s' split: %d samples", args.split, len(dataset))

    result = ModelEvaluator(model, device).evaluate(loader)

    LOGGER.info(
        "[BASELINE: UniversalFakeDetect] split=%s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        args.split, result.metrics.accuracy, result.metrics.precision,
        result.metrics.recall, result.metrics.f1, result.metrics.roc_auc,
    )
    LOGGER.info("[BASELINE: UniversalFakeDetect] confusion_matrix=%s", result.metrics.confusion_matrix)

    ResultWriter.write_predictions(output_dir / f"{args.split}_predictions.csv", result, args.split)
    ResultWriter.write_metrics(output_dir / f"{args.split}_metrics.json", result.metrics)
    LOGGER.info("Wrote predictions and metrics to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
