"""PyTorch Dataset for the real-vs-manipulated binary classifier, shared across model phases
(EfficientNet-B3, ViT-B/16, and any future backbone using the same 224x224 real/fake framing).

Each row of a data/splits/{split}.csv (produced by dataset_split.py) yields two samples:
the untouched data/original/ image (label 0, real) and the SDXL-manipulated data/manipulated/
image (label 1, fake) for the same image_id. Both stay in the split their image_id was
assigned to, so no real/fake pair crosses train/val/test.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import EfficientNetB3Config, HybridFusionConfig, ViTB16Config

ModelConfig = Union[EfficientNetB3Config, ViTB16Config, HybridFusionConfig]

REAL_LABEL = 0
FAKE_LABEL = 1
LABEL_NAMES = {REAL_LABEL: "real", FAKE_LABEL: "fake"}


@dataclass(frozen=True)
class SampleRef:
    image_id: int
    file_name: str
    path: Path
    label: int


class RealFakeDataset(Dataset):
    def __init__(self, split_csv: Path, data_dir: Path, transform: Callable):
        self._samples = self._load_samples(split_csv, data_dir)
        self._transform = transform

    @staticmethod
    def _load_samples(split_csv: Path, data_dir: Path) -> list[SampleRef]:
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        samples: list[SampleRef] = []
        with split_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = int(row["image_id"])
                file_name = row["file_name"]
                samples.append(SampleRef(image_id, file_name, data_dir / "original" / file_name, REAL_LABEL))
                samples.append(SampleRef(image_id, file_name, data_dir / "manipulated" / file_name, FAKE_LABEL))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        sample = self._samples[idx]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
        tensor = self._transform(image)
        return tensor, sample.label, sample.image_id

    def sample_ref(self, idx: int) -> SampleRef:
        return self._samples[idx]


class TransformFactory:
    @staticmethod
    def build(config: ModelConfig, train: bool) -> Callable:
        normalize = transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std)
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        config.image_size, scale=(config.random_resized_crop_scale_min, 1.0)
                    ),
                    transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
                    transforms.RandomRotation(degrees=config.rotation_degrees),
                    transforms.ColorJitter(
                        brightness=config.color_jitter_brightness,
                        contrast=config.color_jitter_contrast,
                        saturation=config.color_jitter_saturation,
                    ),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


def build_dataset(config: ModelConfig, split: str) -> RealFakeDataset:
    if split not in config.split_csv_paths:
        raise ValueError(f"Unknown split '{split}', expected one of {list(config.split_csv_paths)}")
    transform = TransformFactory.build(config, train=(split == "train"))
    return RealFakeDataset(config.split_csv_paths[split], config.data_dir, transform)
