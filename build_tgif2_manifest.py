"""Build the TGIF2 manifest from the extracted tgif-dataset/ folder tree.

Per the TGIF2 acquisition plan (Section 6): parses manipulated-image filenames across the three
shares (tgif, flux, random) into a flat manifest, verifies every manipulated image has a matching
mask file and every coco_id has a matching authentic image in its share, and reports counts needed
before any evaluation code runs against this data (total unique coco_id, per-share counts, TGIF/
FLUX overlap, and any missing pairs - listed, not silently dropped).

Only testing/validation splits exist on disk (training was intentionally never extracted), so this
manifest only covers those two splits.

Usage:
    python build_tgif2_manifest.py
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("tgif-dataset")
MANIFEST_DIR = Path("manifests")

# (source_share, mode, image_dir, mask_dir, filename_regex, gen_model_from_regex)
FR_RE = re.compile(r"^(\d+)_mask_(bbox|segm|random)\.png_([A-Za-z0-9]+)-1024_(\d+)\.png$")
SP_RE = re.compile(r"^(\d+)_mask_(bbox|segm)\.png_ps_mask\.png_([A-Za-z0-9]+)_(\d+)\.png$")
ORIG_RE = re.compile(r"^(\d+)_orig(?:_(\d+))?\.png$")
MASK_RE = re.compile(r"^(\d+)_mask(?:_(bbox|segm|random))?(?:_(\d+))?\.png$")

SHARES = {
    "tgif": {
        "orig_dir": ROOT / "tgif-images" / "orig",
        "mask_dir": ROOT / "tgif-images" / "masks",
        # base_dir contains both "testing/" and "validation/" subdirs - split is inferred from path.
        "manipulated_dirs": {"fr": [("infer", ROOT / "tgif-images" / "sdxl-fr")]},
    },
    "flux": {
        "orig_dir": ROOT / "tgif-flux" / "orig-flux",
        "mask_dir": ROOT / "tgif-flux" / "masks-flux",
        "manipulated_dirs": {
            "fr": [("infer", ROOT / "tgif-flux" / "flux1filldev-fr")],
            "sp": [("infer", ROOT / "tgif-flux" / "flux1filldev-sp")],
        },
    },
    "random": {
        # random-mask evaluation reuses the tgif share's authentic images (same coco_id photos,
        # different mask geometry) - confirmed by inspection: the "tgif ... testing"/"tgif-sdxl-fr
        # validation" folders contain only manipulated files, no *_orig.png of their own.
        "orig_dir": ROOT / "tgif-images" / "orig",
        "mask_dir": {
            "sdxl": ROOT / "tgif-random64_60p" / "masks-sdxl",
            "sd2": ROOT / "tgif-random64_60p" / "masks-sd2",
            "flux": ROOT / "tgif-random64_60p" / "masks-flux",
        },
        # each of these dirs is ALREADY split-specific (no "testing"/"validation" subdir of its
        # own - its immediate children are class folders) - split must be given explicitly, not
        # inferred from path, or every row would get a class name as its "split".
        "manipulated_dirs": {
            "fr": [
                ("testing", ROOT / "tgif-random64_60p" / "tgif  sdxl-fr testing"),
                ("validation", ROOT / "tgif-random64_60p" / "tgif-sdxl-fr validation"),
            ]
        },
    },
}


@dataclass
class ManifestRow:
    coco_id: str
    mask_type: str
    gen_model: str
    var_id: str
    mode: str
    source_share: str
    crop_size: str
    split: str
    file_path: str


def _iter_manipulated(share_name: str, mode: str, dirs: list[tuple[str, Path]]) -> list[ManifestRow]:
    rows = []
    regex = FR_RE if mode == "fr" else SP_RE
    for split_hint, base_dir in dirs:
        if not base_dir.exists():
            continue
        for file_path in base_dir.rglob("*.png"):
            m = regex.match(file_path.name)
            if not m:
                continue
            coco_id, mask_type, gen_model, var_id = m.groups()
            if split_hint == "infer":
                split = file_path.relative_to(base_dir).parts[0]
            else:
                split = split_hint
            rows.append(
                ManifestRow(
                    coco_id=coco_id, mask_type=mask_type, gen_model=gen_model, var_id=var_id,
                    mode=mode, source_share=share_name, crop_size="1024", split=split,
                    file_path=str(file_path),
                )
            )
    return rows


def _unmatched_filenames(share_name: str, mode: str, dirs: list[tuple[str, Path]]) -> list[str]:
    regex = FR_RE if mode == "fr" else SP_RE
    unmatched = []
    for _split_hint, base_dir in dirs:
        if not base_dir.exists():
            continue
        for file_path in base_dir.rglob("*.png"):
            if not regex.match(file_path.name):
                unmatched.append(str(file_path))
    return unmatched


def _orig_coco_ids(orig_dir: Path) -> set[str]:
    ids = set()
    for split_dir in ("testing", "validation"):
        for file_path in (orig_dir / split_dir).rglob("*_orig.png"):
            m = ORIG_RE.match(file_path.name)
            if m:
                ids.add(m.group(1))
    return ids


def _mask_index(mask_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """(coco_id, mask_type) -> list of matching mask file paths, across both splits."""
    index: dict[tuple[str, str], list[Path]] = {}
    for split_dir in ("testing", "validation"):
        d = mask_dir / split_dir
        if not d.exists():
            continue
        for file_path in d.rglob("*.png"):
            m = MASK_RE.match(file_path.name)
            if not m:
                continue
            coco_id, mask_type = m.group(1), m.group(2) or "full"
            index.setdefault((coco_id, mask_type), []).append(file_path)
    return index


def build() -> None:
    MANIFEST_DIR.mkdir(exist_ok=True)
    all_rows: list[ManifestRow] = []
    unmatched_all: list[str] = []
    orig_ids_by_share: dict[str, set[str]] = {}

    for share_name, share in SHARES.items():
        orig_ids_by_share[share_name] = _orig_coco_ids(share["orig_dir"])
        for mode, dirs in share["manipulated_dirs"].items():
            all_rows.extend(_iter_manipulated(share_name, mode, dirs))
            unmatched_all.extend(_unmatched_filenames(share_name, mode, dirs))

    manifest_path = MANIFEST_DIR / "tgif2_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ManifestRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row.__dict__)
    print(f"Wrote {len(all_rows)} manifest rows to {manifest_path}")

    # Split file: coco_id -> train/eval, by share, fixed seed 42, no coco_id crosses a boundary.
    # Not used by any evaluation code yet (zero-shot only), built for consistency with
    # ANALYSIS_PLAN.md as the acquisition plan requested.
    import random

    split_path = MANIFEST_DIR / "tgif2_split.csv"
    with split_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["coco_id", "source_share", "assigned_split"])
        for share_name, ids in orig_ids_by_share.items():
            rng = random.Random(42)
            ids_sorted = sorted(ids)
            rng.shuffle(ids_sorted)
            cutoff = int(0.8 * len(ids_sorted))
            for i, coco_id in enumerate(ids_sorted):
                assigned = "train" if i < cutoff else "eval"
                writer.writerow([coco_id, share_name, assigned])
    print(f"Wrote split file to {split_path}")

    # --- Section 6 report ---
    print("\n=== Section 6 report ===")
    manifest_ids_by_share: dict[str, set[str]] = {}
    for share_name in SHARES:
        manifest_ids_by_share[share_name] = {r.coco_id for r in all_rows if r.source_share == share_name}
        print(f"{share_name}: {len(manifest_ids_by_share[share_name])} unique coco_id with manipulated images "
              f"({sum(1 for r in all_rows if r.source_share == share_name)} manipulated files)")

    print(f"\nTotal unique coco_id across all shares: {len(set().union(*manifest_ids_by_share.values()))}")

    tgif_flux_overlap = manifest_ids_by_share["tgif"] & manifest_ids_by_share["flux"]
    print(f"coco_id present in BOTH tgif and flux shares: {len(tgif_flux_overlap)} "
          f"(of {len(manifest_ids_by_share['tgif'])} tgif / {len(manifest_ids_by_share['flux'])} flux)")

    print(f"\nUnmatched filenames (didn't fit the expected fr/sp grammar): {len(unmatched_all)}")
    for name in unmatched_all[:20]:
        print(f"  UNMATCHED: {name}")

    print("\nManipulated images with no matching mask file:")
    missing_mask_count = 0
    for share_name, share in SHARES.items():
        mask_dir = share["mask_dir"]
        if isinstance(mask_dir, dict):
            mask_indices = {gen: _mask_index(d) for gen, d in mask_dir.items()}
        else:
            shared_index = _mask_index(mask_dir)
            mask_indices = {"_default": shared_index}
        for row in all_rows:
            if row.source_share != share_name:
                continue
            index = mask_indices.get(row.gen_model, mask_indices.get("_default"))
            if index is None or (row.coco_id, row.mask_type) not in index:
                missing_mask_count += 1
                if missing_mask_count <= 20:
                    print(f"  MISSING MASK: {row.file_path} (coco_id={row.coco_id}, mask_type={row.mask_type})")
    print(f"Total manipulated images with no matching mask: {missing_mask_count}")

    print("\ncoco_id present in a share's manipulated set but missing that share's authentic image:")
    missing_orig_count = 0
    for share_name in SHARES:
        missing = manifest_ids_by_share[share_name] - orig_ids_by_share[share_name]
        missing_orig_count += len(missing)
        for coco_id in sorted(missing)[:20]:
            print(f"  MISSING ORIG: share={share_name} coco_id={coco_id}")
    print(f"Total coco_id missing an authentic image in their share: {missing_orig_count}")


if __name__ == "__main__":
    build()
