"""Generate the diffusion-manipulation localization benchmark dataset.

Selects object instances from COCO val2017, builds binary instance masks,
assigns a manipulation (removal/replacement) per a configured object-class
mapping, and runs SDXL inpainting to produce the manipulated image. Existing
samples under data/metadata are treated as already selected and are topped
up with newly selected COCO instances until config.target_count is reached.

Usage:
    python generate_dataset.py --config configs/generate_dataset_config.json
    python generate_dataset.py --config configs/generate_dataset_config.json --dry-run
    python generate_dataset.py --config configs/generate_dataset_config.json --target-count 50

Requires (for actual generation, not for --dry-run):
    pip install torch diffusers transformers accelerate safetensors
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

LOGGER = logging.getLogger("generate_dataset")


def format_image_id(image_id: int) -> str:
    return f"{image_id:012d}"


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


@dataclass(frozen=True)
class GenerationConfig:
    data_dir: Path
    coco_images_dir: Path
    coco_annotations_file: Path
    target_count: int
    model_id: str
    device: str
    torch_dtype: str
    guidance_scale: float
    inference_steps: int
    negative_prompt_default: str
    min_mask_area_ratio: float
    max_mask_area_ratio: float
    manipulation_map: dict[str, dict[str, Any]]
    seed: int
    max_retries: int
    retry_backoff_seconds: float
    dry_run: bool
    enable_attention_slicing: bool
    enable_xformers: bool
    enable_cpu_offload: bool
    log_level: str
    manifest_filename: str
    csv_summary_filename: str

    @classmethod
    def from_file(cls, config_path: Path, overrides: Optional[dict[str, Any]] = None) -> "GenerationConfig":
        with config_path.open("r") as f:
            raw: dict[str, Any] = json.load(f)
        for key, value in (overrides or {}).items():
            if value is not None:
                raw[key] = value

        return cls(
            data_dir=Path(raw["data_dir"]).expanduser().resolve(),
            coco_images_dir=Path(raw["coco_images_dir"]).expanduser().resolve(),
            coco_annotations_file=Path(raw["coco_annotations_file"]).expanduser().resolve(),
            target_count=int(raw["target_count"]),
            model_id=raw["model_id"],
            device=raw["device"],
            torch_dtype=raw["torch_dtype"],
            guidance_scale=float(raw["guidance_scale"]),
            inference_steps=int(raw["inference_steps"]),
            negative_prompt_default=raw.get("negative_prompt_default", ""),
            min_mask_area_ratio=float(raw["min_mask_area_ratio"]),
            max_mask_area_ratio=float(raw["max_mask_area_ratio"]),
            manipulation_map=raw["manipulation_map"],
            seed=int(raw["seed"]),
            max_retries=int(raw["max_retries"]),
            retry_backoff_seconds=float(raw["retry_backoff_seconds"]),
            dry_run=bool(raw.get("dry_run", False)),
            enable_attention_slicing=bool(raw.get("enable_attention_slicing", True)),
            enable_xformers=bool(raw.get("enable_xformers", True)),
            enable_cpu_offload=bool(raw.get("enable_cpu_offload", True)),
            log_level=raw.get("log_level", "INFO"),
            manifest_filename=raw.get("manifest_filename", "manifest.json"),
            csv_summary_filename=raw.get("csv_summary_filename", "dataset_summary.csv"),
        )


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    original_dir: Path
    masks_dir: Path
    manipulated_dir: Path
    metadata_dir: Path

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "DatasetPaths":
        paths = cls(
            data_dir=data_dir,
            original_dir=data_dir / "original",
            masks_dir=data_dir / "masks",
            manipulated_dir=data_dir / "manipulated",
            metadata_dir=data_dir / "metadata",
        )
        for directory in (paths.original_dir, paths.masks_dir, paths.manipulated_dir, paths.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass
class SampleMetadata:
    image_id: int
    file_name: str
    source_dataset: str
    object_class: str
    mask_area: int
    mask_area_ratio: float
    manipulation_type: str
    prompt: str
    negative_prompt: str
    diffusion_model: str
    guidance_scale: float
    inference_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CocoInstance:
    image_id: int
    file_name: str
    category_name: str
    segmentation: list[list[float]]
    area: float
    bbox: list[float]
    image_width: int
    image_height: int


class ChecksumUtil:
    _CHUNK_SIZE = 1 << 20

    @classmethod
    def sha256_file(cls, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(cls._CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()


class AtomicWriter:
    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(path)

    @staticmethod
    def write_image(path: Path, image: Image.Image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        fmt = "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
        save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
        image.save(tmp_path, format=fmt, **save_kwargs)
        tmp_path.replace(path)

    @staticmethod
    def copy_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp_path)
        tmp_path.replace(dst)


class SampleValidator:
    @staticmethod
    def validate_image(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception as exc:
            LOGGER.error("Validation failed for %s: %s", path, exc)
            return False


class ManifestStore:
    """Tracks per-sample generation status to support resume across runs."""

    def __init__(self, path: Path):
        self._path = path
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = {}
            return
        try:
            self._entries = json.loads(self._path.read_text())
        except json.JSONDecodeError as exc:
            LOGGER.error("Manifest at %s is corrupt (%s); starting a fresh manifest", self._path, exc)
            self._entries = {}

    def get(self, image_id: int) -> Optional[dict[str, Any]]:
        return self._entries.get(str(image_id))

    def update(self, image_id: int, **fields: Any) -> None:
        key = str(image_id)
        entry = self._entries.get(key, {"image_id": image_id, "attempts": 0})
        entry.update(fields)
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._entries[key] = entry
        AtomicWriter.write_json(self._path, self._entries)


class PromptFactory:
    def __init__(self, manipulation_map: dict[str, dict[str, Any]], negative_prompt_default: str):
        self._manipulation_map = manipulation_map
        self._negative_prompt_default = negative_prompt_default

    def build(self, object_class: str) -> tuple[str, str, str]:
        entry = self._manipulation_map.get(object_class)
        if entry is None:
            raise KeyError(f"No manipulation mapping configured for object class '{object_class}'")
        manipulation_type = entry["type"]
        if manipulation_type == "removal":
            prompt = f"remove the {object_class}"
        elif manipulation_type == "replacement":
            prompt = f"replace {object_class} with {entry['target']}"
        else:
            raise ValueError(f"Unknown manipulation type '{manipulation_type}' for class '{object_class}'")
        return manipulation_type, prompt, self._negative_prompt_default


class MaskRenderer:
    @staticmethod
    def render(instance: CocoInstance) -> Image.Image:
        mask = Image.new("L", (instance.image_width, instance.image_height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in instance.segmentation:
            if len(polygon) < 6:
                continue
            points = list(zip(polygon[0::2], polygon[1::2]))
            draw.polygon(points, fill=255)
        return mask


class CocoIndex:
    """Indexes COCO instance annotations restricted to a set of target classes."""

    def __init__(self, annotations_file: Path, target_classes: set[str]):
        self._instances: list[CocoInstance] = []
        self._load(annotations_file, target_classes)

    def _load(self, annotations_file: Path, target_classes: set[str]) -> None:
        LOGGER.info("Loading COCO annotations from %s", annotations_file)
        with annotations_file.open("r") as f:
            data = json.load(f)

        categories = {c["id"]: c["name"] for c in data["categories"]}
        images = {im["id"]: im for im in data["images"]}
        target_category_ids = {cid for cid, name in categories.items() if name in target_classes}

        for ann in data["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            if ann["category_id"] not in target_category_ids:
                continue
            segmentation = ann.get("segmentation")
            if not isinstance(segmentation, list) or not segmentation:
                continue
            image_info = images.get(ann["image_id"])
            if image_info is None:
                continue
            self._instances.append(
                CocoInstance(
                    image_id=ann["image_id"],
                    file_name=image_info["file_name"],
                    category_name=categories[ann["category_id"]],
                    segmentation=segmentation,
                    area=float(ann["area"]),
                    bbox=ann["bbox"],
                    image_width=image_info["width"],
                    image_height=image_info["height"],
                )
            )

        LOGGER.info(
            "Indexed %d candidate instances across %d target classes",
            len(self._instances), len(target_category_ids),
        )

    def shuffled_candidates(self, rng: random.Random) -> list[CocoInstance]:
        candidates = list(self._instances)
        rng.shuffle(candidates)
        return candidates


def retry_with_backoff(func: Callable[[], Image.Image], max_retries: int, backoff_seconds: float) -> Image.Image:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - broad by design, retried and logged
            last_exc = exc
            LOGGER.warning("Attempt %d/%d raised %s: %s", attempt, max_retries, type(exc).__name__, exc)
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    assert last_exc is not None
    raise last_exc


class SDXLInpaintGenerator:
    """Lazy-loading wrapper around diffusers' SDXL inpainting pipeline."""

    def __init__(self, config: GenerationConfig):
        self._config = config
        self._device = config.device
        self._pipeline: Any = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionXLInpaintPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers/torch are required for SDXL generation. Install with: "
                "pip install torch diffusers transformers accelerate safetensors"
            ) from exc

        if self._device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available; falling back to CPU")
            self._device = "cpu"

        dtype = getattr(torch, self._config.torch_dtype) if self._device == "cuda" else torch.float32
        LOGGER.info("Loading SDXL inpainting pipeline '%s' on %s (%s)", self._config.model_id, self._device, dtype)
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            self._config.model_id, torch_dtype=dtype, use_safetensors=True,
        )

        if self._config.enable_cpu_offload and self._device == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(self._device)

        if self._config.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        if self._config.enable_xformers and self._device == "cuda":
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception as exc:  # noqa: BLE001 - optional acceleration, safe to skip
                LOGGER.warning("xformers memory efficient attention unavailable: %s", exc)

        self._pipeline = pipeline

    @staticmethod
    def _pad_to_multiple(image: Image.Image, multiple: int = 8) -> tuple[Image.Image, int, int]:
        # SDXL's VAE requires dimensions divisible by 8; COCO images rarely are.
        width, height = image.size
        padded_width = math.ceil(width / multiple) * multiple
        padded_height = math.ceil(height / multiple) * multiple
        if (padded_width, padded_height) == (width, height):
            return image, width, height
        padded = Image.new(image.mode, (padded_width, padded_height))
        padded.paste(image, (0, 0))
        return padded, width, height

    def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        inference_steps: int,
        seed: int,
    ) -> Image.Image:
        self._ensure_loaded()
        import torch

        padded_image, orig_width, orig_height = self._pad_to_multiple(image)
        padded_mask, _, _ = self._pad_to_multiple(mask)

        generator_device = "cpu" if self._config.enable_cpu_offload and self._device == "cuda" else self._device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=padded_image,
            mask_image=padded_mask,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            width=padded_image.width,
            height=padded_image.height,
            generator=generator,
        )
        result = output.images[0]
        if (result.width, result.height) != (orig_width, orig_height):
            result = result.crop((0, 0, orig_width, orig_height))
        return result

    def cleanup(self) -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass


class DatasetGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.paths = DatasetPaths.from_data_dir(config.data_dir)
        self.manifest = ManifestStore(config.data_dir / config.manifest_filename)
        self.prompt_factory = PromptFactory(config.manipulation_map, config.negative_prompt_default)
        self.generator = SDXLInpaintGenerator(config)

    def run(self) -> None:
        existing_records = self._load_existing_metadata()
        existing_ids = {m.image_id for m in existing_records}
        needed = max(0, self.config.target_count - len(existing_records))
        LOGGER.info(
            "Existing metadata records: %d | target: %d | selecting %d new sample(s)",
            len(existing_records), self.config.target_count, needed,
        )

        new_records = self._select_new_candidates(existing_ids, needed)
        all_records = existing_records + new_records
        LOGGER.info("Total records to process: %d", len(all_records))

        status_counts: dict[str, int] = {"completed": 0, "failed": 0, "skipped": 0, "dry_run": 0}
        for meta in tqdm(all_records, desc="Generating manipulated images", unit="sample"):
            outcome = self._process_one(meta)
            status_counts[outcome] = status_counts.get(outcome, 0) + 1

        self._write_csv_summary(all_records)
        self.generator.cleanup()
        LOGGER.info(
            "Generation complete. completed=%d failed=%d skipped=%d dry_run=%d (total=%d)",
            status_counts["completed"], status_counts["failed"],
            status_counts["skipped"], status_counts["dry_run"], len(all_records),
        )

    def _load_existing_metadata(self) -> list[SampleMetadata]:
        records: list[SampleMetadata] = []
        for path in sorted(self.paths.metadata_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                records.append(SampleMetadata(**data))
            except Exception as exc:  # noqa: BLE001 - a single bad file shouldn't abort the run
                LOGGER.error("Failed to load metadata %s: %s", path, exc)
        return records

    def _select_new_candidates(self, existing_ids: set[int], needed: int) -> list[SampleMetadata]:
        if needed <= 0:
            return []

        coco_index = CocoIndex(self.config.coco_annotations_file, target_classes=set(self.config.manipulation_map))
        rng = random.Random(self.config.seed)
        candidates = coco_index.shuffled_candidates(rng)

        used_image_ids = set(existing_ids)
        new_records: list[SampleMetadata] = []

        for instance in candidates:
            if len(new_records) >= needed:
                break
            if instance.image_id in used_image_ids:
                continue

            area_ratio = instance.area / (instance.image_width * instance.image_height)
            if not (self.config.min_mask_area_ratio <= area_ratio <= self.config.max_mask_area_ratio):
                continue

            source_image_path = self.config.coco_images_dir / instance.file_name
            if not source_image_path.exists():
                LOGGER.warning("Source COCO image missing, skipping: %s", source_image_path)
                continue

            mask_image = MaskRenderer.render(instance)
            mask_area_pixels = int(np.asarray(mask_image).astype(bool).sum())
            if mask_area_pixels == 0:
                continue

            image_id_str = format_image_id(instance.image_id)
            manipulation_type, prompt, negative_prompt = self.prompt_factory.build(instance.category_name)
            image_area = instance.image_width * instance.image_height
            metadata = SampleMetadata(
                image_id=instance.image_id,
                file_name=f"{image_id_str}.jpg",
                source_dataset="COCO",
                object_class=instance.category_name,
                mask_area=mask_area_pixels,
                mask_area_ratio=round(mask_area_pixels / image_area, 4),
                manipulation_type=manipulation_type,
                prompt=prompt,
                negative_prompt=negative_prompt,
                diffusion_model="SDXL",
                guidance_scale=self.config.guidance_scale,
                inference_steps=self.config.inference_steps,
            )

            AtomicWriter.copy_file(source_image_path, self.paths.original_dir / metadata.file_name)
            AtomicWriter.write_image(self.paths.masks_dir / f"{image_id_str}_mask.png", mask_image)
            AtomicWriter.write_json(self.paths.metadata_dir / f"{image_id_str}.json", metadata.to_dict())

            used_image_ids.add(instance.image_id)
            new_records.append(metadata)

        if len(new_records) < needed:
            LOGGER.warning("Only found %d/%d new candidates matching the configured filters", len(new_records), needed)
        return new_records

    def _process_one(self, meta: SampleMetadata) -> str:
        image_id_str = format_image_id(meta.image_id)
        original_path = self.paths.original_dir / meta.file_name
        mask_path = self.paths.masks_dir / f"{image_id_str}_mask.png"
        manipulated_path = self.paths.manipulated_dir / meta.file_name

        entry = self.manifest.get(meta.image_id)
        if entry and entry.get("status") == "completed":
            if manipulated_path.exists() and ChecksumUtil.sha256_file(manipulated_path) == entry.get("sha256"):
                return "skipped"
            LOGGER.warning("Manifest marks %s completed but output is missing/corrupt; regenerating", image_id_str)

        if not SampleValidator.validate_image(original_path) or not SampleValidator.validate_image(mask_path):
            LOGGER.error("Invalid or missing input for sample %s (original/mask)", image_id_str)
            self.manifest.update(
                meta.image_id, status="failed", attempts=(entry or {}).get("attempts", 0),
                error="missing_or_invalid_input", sha256=None,
            )
            return "failed"

        if self.config.dry_run:
            LOGGER.info("[DRY RUN] Would generate %s | type=%s | prompt=%r", image_id_str, meta.manipulation_type, meta.prompt)
            self.manifest.update(meta.image_id, status="dry_run", attempts=(entry or {}).get("attempts", 0), error=None, sha256=None)
            return "dry_run"

        attempts_holder = {"count": (entry or {}).get("attempts", 0)}

        def _attempt() -> Image.Image:
            attempts_holder["count"] += 1
            with Image.open(original_path) as im, Image.open(mask_path) as mk:
                image = im.convert("RGB")
                mask = mk.convert("L")
            effective_seed = (self.config.seed + meta.image_id) % (2**31 - 1)
            return self.generator.generate(
                image=image,
                mask=mask,
                prompt=meta.prompt,
                negative_prompt=meta.negative_prompt,
                guidance_scale=meta.guidance_scale,
                inference_steps=meta.inference_steps,
                seed=effective_seed,
            )

        try:
            result_image = retry_with_backoff(_attempt, self.config.max_retries, self.config.retry_backoff_seconds)
        except Exception as exc:  # noqa: BLE001 - failure is recorded and the run continues
            LOGGER.error("Sample %s failed after %d attempt(s): %s", image_id_str, attempts_holder["count"], exc)
            self.manifest.update(
                meta.image_id, status="failed", attempts=attempts_holder["count"], error=str(exc), sha256=None,
            )
            return "failed"
        finally:
            self.generator.cleanup()

        AtomicWriter.write_image(manipulated_path, result_image)
        checksum = ChecksumUtil.sha256_file(manipulated_path)
        self.manifest.update(meta.image_id, status="completed", attempts=attempts_holder["count"], error=None, sha256=checksum)
        return "completed"

    def _write_csv_summary(self, records: list[SampleMetadata]) -> None:
        rows = []
        for meta in records:
            entry = self.manifest.get(meta.image_id) or {}
            row = meta.to_dict()
            row["status"] = entry.get("status", "pending")
            row["sha256"] = entry.get("sha256", "")
            row["attempts"] = entry.get("attempts", 0)
            row["error"] = entry.get("error", "")
            rows.append(row)

        if not rows:
            return

        csv_path = self.config.data_dir / self.config.csv_summary_filename
        tmp_path = csv_path.with_name(csv_path.name + ".tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(csv_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/generate_dataset_config.json"))
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    overrides = {
        "dry_run": args.dry_run,
        "target_count": args.target_count,
        "data_dir": args.data_dir,
        "log_level": args.log_level,
    }
    config = GenerationConfig.from_file(args.config, overrides)
    configure_logging(config.log_level, config.data_dir / "logs" / "generate_dataset.log")
    LOGGER.info("Loaded configuration from %s", args.config)

    try:
        DatasetGenerator(config).run()
    except Exception:  # noqa: BLE001 - top-level guard so partial progress is already persisted
        LOGGER.error("Fatal error during dataset generation:\n%s", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
