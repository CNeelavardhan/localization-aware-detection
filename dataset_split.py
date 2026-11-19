"""Split the diffusion-manipulation localization benchmark into train/val/test.

Reads data/dataset_summary.csv (produced by generate_dataset.py), keeps only
samples whose status is in config.included_statuses (by default only samples
with a real, verified manipulated image: "completed"), validates that the
original/mask/manipulated files exist and are readable, and assigns each
sample to a split using a deterministic stratified split keyed on
config.stratify_columns (default: object_class). Writes per-split CSVs, a
splits.json manifest, and a split_summary.csv with per-group/per-split counts.

Usage:
    python dataset_split.py --config configs/dataset_split_config.json
    python dataset_split.py --config configs/dataset_split_config.json --dry-run
    python dataset_split.py --config configs/dataset_split_config.json \
        --included-statuses completed dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Optional

from PIL import Image

LOGGER = logging.getLogger("dataset_split")

_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


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
class SplitConfig:
    data_dir: Path
    summary_csv_filename: str
    included_statuses: tuple[str, ...]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    stratify_columns: tuple[str, ...]
    seed: int
    min_group_size_for_full_stratification: int
    dry_run: bool
    log_level: str
    output_dir: Path
    splits_manifest_filename: str
    split_summary_filename: str

    @classmethod
    def from_file(cls, config_path: Path, overrides: Optional[dict[str, Any]] = None) -> "SplitConfig":
        with config_path.open("r") as f:
            raw: dict[str, Any] = json.load(f)
        for key, value in (overrides or {}).items():
            if value is not None:
                raw[key] = value

        data_dir = Path(raw["data_dir"]).expanduser().resolve()
        train_ratio = float(raw["train_ratio"])
        val_ratio = float(raw["val_ratio"])
        test_ratio = float(raw["test_ratio"])
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")

        included_statuses = raw["included_statuses"]
        if isinstance(included_statuses, str):
            included_statuses = [included_statuses]
        stratify_columns = raw["stratify_columns"]
        if isinstance(stratify_columns, str):
            stratify_columns = [stratify_columns]

        return cls(
            data_dir=data_dir,
            summary_csv_filename=raw["summary_csv_filename"],
            included_statuses=tuple(included_statuses),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_columns=tuple(stratify_columns),
            seed=int(raw["seed"]),
            min_group_size_for_full_stratification=int(raw["min_group_size_for_full_stratification"]),
            dry_run=bool(raw.get("dry_run", False)),
            log_level=raw.get("log_level", "INFO"),
            output_dir=Path(raw["output_dir"]).expanduser().resolve(),
            splits_manifest_filename=raw.get("splits_manifest_filename", "splits.json"),
            split_summary_filename=raw.get("split_summary_filename", "split_summary.csv"),
        )

    @property
    def ratios(self) -> dict[str, float]:
        return {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio}


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    original_dir: Path
    masks_dir: Path
    manipulated_dir: Path

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "DatasetPaths":
        return cls(
            data_dir=data_dir,
            original_dir=data_dir / "original",
            masks_dir=data_dir / "masks",
            manipulated_dir=data_dir / "manipulated",
        )


@dataclass(frozen=True)
class SampleRecord:
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
    status: str
    sha256: str
    attempts: int
    error: str
    fields: dict[str, Any] = field(repr=False, compare=False)

    def stratify_key(self, columns: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(str(getattr(self, column)) for column in columns)


class SampleValidator:
    @staticmethod
    def validate_image(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception as exc:  # noqa: BLE001 - any decode failure means invalid
            LOGGER.error("Validation failed for %s: %s", path, exc)
            return False


class AtomicWriter:
    @staticmethod
    def write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(path)

    @staticmethod
    def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)


class SummaryCsvLoader:
    """Loads sample records from the dataset_summary.csv produced by generate_dataset.py."""

    _INT_FIELDS = ("image_id", "mask_area", "inference_steps", "attempts")
    _FLOAT_FIELDS = ("mask_area_ratio", "guidance_scale")

    def load(self, csv_path: Path) -> list[SampleRecord]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset summary CSV not found: {csv_path}")

        records: list[SampleRecord] = []
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    records.append(self._parse_row(row))
                except (KeyError, ValueError) as exc:
                    LOGGER.error("Skipping malformed row %d in %s: %s", row_num, csv_path, exc)
        LOGGER.info("Loaded %d sample record(s) from %s", len(records), csv_path)
        return records

    def _parse_row(self, row: dict[str, str]) -> SampleRecord:
        parsed: dict[str, Any] = dict(row)
        for key in self._INT_FIELDS:
            parsed[key] = int(row[key]) if row.get(key) not in (None, "") else 0
        for key in self._FLOAT_FIELDS:
            parsed[key] = float(row[key]) if row.get(key) not in (None, "") else 0.0
        return SampleRecord(
            image_id=parsed["image_id"],
            file_name=row["file_name"],
            source_dataset=row["source_dataset"],
            object_class=row["object_class"],
            mask_area=parsed["mask_area"],
            mask_area_ratio=parsed["mask_area_ratio"],
            manipulation_type=row["manipulation_type"],
            prompt=row["prompt"],
            negative_prompt=row.get("negative_prompt", ""),
            diffusion_model=row["diffusion_model"],
            guidance_scale=parsed["guidance_scale"],
            inference_steps=parsed["inference_steps"],
            status=row.get("status", ""),
            sha256=row.get("sha256", ""),
            attempts=parsed["attempts"],
            error=row.get("error", ""),
            fields=dict(row),
        )


class RecordValidator:
    """Confirms the original/mask/manipulated files for a sample actually exist and decode."""

    def __init__(self, paths: DatasetPaths):
        self._paths = paths

    def is_valid(self, record: SampleRecord) -> bool:
        image_id_str = f"{record.image_id:012d}"
        original_path = self._paths.original_dir / record.file_name
        mask_path = self._paths.masks_dir / f"{image_id_str}_mask.png"
        manipulated_path = self._paths.manipulated_dir / record.file_name

        for path in (original_path, mask_path, manipulated_path):
            if not SampleValidator.validate_image(path):
                LOGGER.warning("Sample %s excluded from split: invalid/missing %s", record.image_id, path)
                return False
        return True


class StratifiedSplitter:
    """Deterministically assigns records to train/val/test, stratified by group key."""

    def __init__(self, ratios: dict[str, float], seed: int, min_group_size: int):
        self._ratios = ratios
        self._seed = seed
        self._min_group_size = min_group_size

    def split(
        self, records: list[SampleRecord], stratify_columns: tuple[str, ...],
    ) -> dict[int, str]:
        groups: dict[tuple[str, ...], list[SampleRecord]] = {}
        for record in records:
            key = record.stratify_key(stratify_columns)
            groups.setdefault(key, []).append(record)

        assignment: dict[int, str] = {}
        for key in sorted(groups.keys()):
            group_records = sorted(groups[key], key=lambda r: r.image_id)
            rng = Random(f"{self._seed}:{key}")
            rng.shuffle(group_records)

            if len(group_records) < self._min_group_size:
                LOGGER.info(
                    "Group %s has only %d sample(s) (< min_group_size=%d); assigning all to train",
                    key, len(group_records), self._min_group_size,
                )
                for record in group_records:
                    assignment[record.image_id] = "train"
                continue

            counts = self._largest_remainder_counts(len(group_records))
            cursor = 0
            for split_name in _SPLIT_NAMES:
                count = counts[split_name]
                for record in group_records[cursor:cursor + count]:
                    assignment[record.image_id] = split_name
                cursor += count

        return assignment

    def _largest_remainder_counts(self, total: int) -> dict[str, int]:
        raw = {name: total * self._ratios[name] for name in _SPLIT_NAMES}
        floored = {name: int(raw[name]) for name in _SPLIT_NAMES}
        remainder = total - sum(floored.values())

        remainders = sorted(_SPLIT_NAMES, key=lambda name: raw[name] - floored[name], reverse=True)
        for name in remainders[:remainder]:
            floored[name] += 1
        return floored


class DatasetSplitter:
    def __init__(self, config: SplitConfig):
        self.config = config
        self.paths = DatasetPaths.from_data_dir(config.data_dir)
        self.loader = SummaryCsvLoader()
        self.record_validator = RecordValidator(self.paths)
        self.splitter = StratifiedSplitter(config.ratios, config.seed, config.min_group_size_for_full_stratification)

    def run(self) -> int:
        csv_path = self.config.data_dir / self.config.summary_csv_filename
        all_records = self.loader.load(csv_path)

        eligible = [r for r in all_records if r.status in self.config.included_statuses]
        LOGGER.info(
            "Eligible by status %s: %d/%d record(s)",
            self.config.included_statuses, len(eligible), len(all_records),
        )

        valid_records = [r for r in eligible if self.record_validator.is_valid(r)]
        invalid_count = len(eligible) - len(valid_records)
        if invalid_count:
            LOGGER.warning("Excluded %d eligible sample(s) that failed file validation", invalid_count)

        if not valid_records:
            LOGGER.error(
                "No valid samples available to split (checked %d eligible record(s)). "
                "Nothing to write.", len(eligible),
            )
            return 1

        assignment = self.splitter.split(valid_records, self.config.stratify_columns)
        records_by_split: dict[str, list[SampleRecord]] = {name: [] for name in _SPLIT_NAMES}
        for record in valid_records:
            records_by_split[assignment[record.image_id]].append(record)

        for split_name in _SPLIT_NAMES:
            LOGGER.info("Split '%s': %d sample(s)", split_name, len(records_by_split[split_name]))

        summary_rows = self._build_summary_rows(valid_records, assignment)

        if self.config.dry_run:
            LOGGER.info("[DRY RUN] Would write split CSVs, manifest, and summary to %s", self.config.output_dir)
            return 0

        self._write_split_csvs(records_by_split)
        self._write_manifest(assignment, valid_records)
        self._write_summary(summary_rows)
        LOGGER.info("Dataset split complete. Output written to %s", self.config.output_dir)
        return 0

    def _build_summary_rows(
        self, records: list[SampleRecord], assignment: dict[int, str],
    ) -> list[dict[str, Any]]:
        counts: dict[tuple[str, ...], dict[str, int]] = {}
        for record in records:
            key = record.stratify_key(self.config.stratify_columns)
            split_name = assignment[record.image_id]
            counts.setdefault(key, {name: 0 for name in _SPLIT_NAMES})
            counts[key][split_name] += 1

        rows: list[dict[str, Any]] = []
        for key in sorted(counts.keys()):
            row: dict[str, Any] = dict(zip(self.config.stratify_columns, key))
            row.update(counts[key])
            row["total"] = sum(counts[key].values())
            rows.append(row)

        totals_row: dict[str, Any] = {col: "TOTAL" for col in self.config.stratify_columns}
        totals_row.update({name: sum(r[name] for r in rows) for name in _SPLIT_NAMES})
        totals_row["total"] = sum(totals_row[name] for name in _SPLIT_NAMES)
        rows.append(totals_row)
        return rows

    def _write_split_csvs(self, records_by_split: dict[str, list[SampleRecord]]) -> None:
        fieldnames = [
            "image_id", "file_name", "source_dataset", "object_class", "mask_area",
            "mask_area_ratio", "manipulation_type", "prompt", "negative_prompt",
            "diffusion_model", "guidance_scale", "inference_steps", "sha256",
        ]
        for split_name, records in records_by_split.items():
            rows = [
                {name: getattr(record, name) for name in fieldnames}
                for record in sorted(records, key=lambda r: r.image_id)
            ]
            AtomicWriter.write_csv(self.config.output_dir / f"{split_name}.csv", rows, fieldnames)

    def _write_manifest(self, assignment: dict[int, str], records: list[SampleRecord]) -> None:
        manifest = {
            str(record.image_id): {
                "image_id": record.image_id,
                "split": assignment[record.image_id],
                "object_class": record.object_class,
                "manipulation_type": record.manipulation_type,
            }
            for record in records
        }
        AtomicWriter.write_json(self.config.output_dir / self.config.splits_manifest_filename, manifest)

    def _write_summary(self, rows: list[dict[str, Any]]) -> None:
        fieldnames = list(self.config.stratify_columns) + list(_SPLIT_NAMES) + ["total"]
        AtomicWriter.write_csv(self.config.output_dir / self.config.split_summary_filename, rows, fieldnames)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_split_config.json"))
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--included-statuses", type=str, nargs="+", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    overrides = {
        "dry_run": args.dry_run,
        "data_dir": args.data_dir,
        "log_level": args.log_level,
        "included_statuses": args.included_statuses,
    }
    config = SplitConfig.from_file(args.config, overrides)
    configure_logging(config.log_level, config.data_dir / "logs" / "dataset_split.log")
    LOGGER.info("Loaded configuration from %s", args.config)

    try:
        return DatasetSplitter(config).run()
    except Exception:  # noqa: BLE001 - top-level guard, error already logged with traceback
        LOGGER.error("Fatal error during dataset split:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
