#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare BrainMetShare into the flat BrainMetShare-prepared layout.

Source layout:
    BrainMetShare/brainmetshare-3/{train,test}/Mets_xxx/*.nii.gz

Prepared layout:
    BrainMetShare-prepared/Mets_xxx/{bravo,t1_pre,flair,t1_gd,seg}.nii.gz

The default behavior is conservative: if the destination already exists and is not empty, 
the script exits unless --dry-run, --force-existing, --overwrite, or --clean is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SRC = Path("/home/dhm_41310/hdd/trzhang/datasets/BrainMetShare")
DEFAULT_DST = Path("/home/dhm_41310/hdd/trzhang/datasets/BrainMetShare-prepared")
DATASET_SUBDIR = "brainmetshare-3"
MODALITIES = ("bravo.nii.gz", "t1_pre.nii.gz", "flair.nii.gz", "t1_gd.nii.gz")
SEGMENTATION = "seg.nii.gz"
PNG_GAMMA = 1.4

@dataclass(frozen=True)
class CaseRecord:
    split: str
    case_id: str
    source_dir: Path
    output_dir: Path
    has_seg: bool
    missing_files: tuple[str, ...]


def natural_key(path: Path) -> tuple:
    text = path.name
    parts: list[object] = []
    buf = ""
    is_digit = False
    for ch in text:
        if ch.isdigit() != is_digit and buf:
            parts.append(int(buf) if is_digit else buf.lower())
            buf = ""
        buf += ch
        is_digit = ch.isdigit()
    if buf:
        parts.append(int(buf) if is_digit else buf.lower())
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BrainMetShare as /BrainMetShare-prepared/Mets_xxx/*.nii.gz."
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"Source root. Default: {DEFAULT_SRC}")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help=f"Output root. Default: {DEFAULT_DST}")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    parser.add_argument(
        "--force-existing",
        action="store_true",
        help="Allow writing into an existing non-empty destination without replacing existing files.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete destination before preparing.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report only; do not write anything.")
    parser.add_argument("--generate-png", action="store_true", help="Generate axial PNG previews for labeled cases from t1_pre.nii.gz.")
    parser.add_argument("--num-png", type=int, default=8, help="Number of PNG previews per case. Default: 8.")
    parser.add_argument("--png-size", type=int, default=448, help="Output PNG width/height. Default: 448.")
    parser.add_argument("--png-subdir", default="vlm_png_axial", help="PNG preview subdirectory name.")
    return parser.parse_args()


def iter_cases(src_root: Path) -> Iterable[tuple[str, Path]]:
    dataset_root = src_root / DATASET_SUBDIR
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")
    for split in ("train", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for case_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=natural_key):
            yield split, case_dir


def inspect_case(split: str, case_dir: Path, dst_root: Path) -> CaseRecord:
    missing = tuple(name for name in MODALITIES if not (case_dir / name).is_file())
    return CaseRecord(
        split=split,
        case_id=case_dir.name,
        source_dir=case_dir,
        output_dir=dst_root / case_dir.name,
        has_seg=(case_dir / SEGMENTATION).is_file(),
        missing_files=missing,
    )


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_case(record: CaseRecord, overwrite: bool) -> None:
    if record.missing_files:
        raise FileNotFoundError(f"{record.case_id} is missing: {', '.join(record.missing_files)}")
    record.output_dir.mkdir(parents=True, exist_ok=True)
    for name in MODALITIES:
        copy_file(record.source_dir / name, record.output_dir / name, overwrite)
    if record.has_seg:
        copy_file(record.source_dir / SEGMENTATION, record.output_dir / SEGMENTATION, overwrite)


def normalize_slice(slice_2d):
    import numpy as np

    arr = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[finite]
    lo, hi = np.percentile(vals, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    arr = np.power(arr, PNG_GAMMA)
    return (arr * 255.0).astype(np.uint8)


def axial_slice_indices(z_count: int, num_png: int) -> list[int]:
    import numpy as np

    if num_png <= 0:
        return []
    if z_count <= 0:
        raise ValueError("Empty z dimension")
    if num_png == 1:
        return [z_count // 2]
    start = int(round((z_count - 1) * 0.10))
    stop = int(round((z_count - 1) * 0.90))
    return [int(z) for z in np.linspace(start, stop, num_png).round().astype(int).tolist()]


def generate_axial_pngs(case: CaseRecord, png_subdir: str, num_png: int, png_size: int, overwrite: bool) -> None:
    if num_png <= 0:
        return
    try:
        import nibabel as nib
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("--generate-png requires nibabel, numpy, and pillow") from exc

    if png_size <= 0:
        raise ValueError("--png-size must be positive")

    png_dir = case.output_dir / png_subdir
    done_marker = png_dir / f".done_BrainMetShare_{num_png}_{png_size}"
    if png_dir.exists() and overwrite:
        shutil.rmtree(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)

    image_path = case.output_dir / "t1_pre.nii.gz"
    if not image_path.is_file():
        image_path = case.output_dir / "t1_gd.nii.gz"
    img = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    if img.ndim > 3:
        img = np.squeeze(img)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D t1_gd image for {case.case_id}, got shape {img.shape}")

    indices = axial_slice_indices(img.shape[2], num_png)
    expected_pngs = [png_dir / f"z{z:03d}.png" for z in indices]
    if done_marker.exists() and all(p.exists() for p in expected_pngs) and not overwrite:
        return

    for old_png in png_dir.glob("*.png"):
        old_png.unlink()

    for z, out_path in zip(indices, expected_pngs):
        arr = normalize_slice(img[:, :, int(z)])
        image = Image.fromarray(arr, mode="L").resize((png_size, png_size), Image.Resampling.BILINEAR)
        image.convert("RGB").save(out_path)
    done_marker.touch()


def write_manifest(records: list[CaseRecord], dst_root: Path) -> None:
    with (dst_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "dataset": "BrainMetShare",
                "split": rec.split,
                "case_id": rec.case_id,
                "case_dir": str(rec.output_dir),
                "img_path": str(rec.output_dir / "t1_gd.nii.gz"),
                "seg_path": str(rec.output_dir / SEGMENTATION) if rec.has_seg else "",
                "has_seg": rec.has_seg,
                "modalities": list(MODALITIES),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with (dst_root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ("dataset", "split", "case_id", "case_dir", "img_path", "seg_path", "has_seg")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "dataset": "BrainMetShare",
                    "split": rec.split,
                    "case_id": rec.case_id,
                    "case_dir": rec.output_dir,
                    "img_path": rec.output_dir / "t1_gd.nii.gz",
                    "seg_path": rec.output_dir / SEGMENTATION if rec.has_seg else "",
                    "has_seg": rec.has_seg,
                }
            )


def print_summary(records: list[CaseRecord], dst_root: Path, dry_run: bool) -> None:
    split_counts: dict[str, int] = {}
    labeled_counts: dict[str, int] = {}
    existing_case_dirs = 0
    existing_files = 0
    expected_files = 0
    for rec in records:
        split_counts[rec.split] = split_counts.get(rec.split, 0) + 1
        if rec.has_seg:
            labeled_counts[rec.split] = labeled_counts.get(rec.split, 0) + 1
        if rec.output_dir.exists():
            existing_case_dirs += 1
        names = list(MODALITIES) + ([SEGMENTATION] if rec.has_seg else [])
        expected_files += len(names)
        existing_files += sum(1 for name in names if (rec.output_dir / name).exists())

    print(f"{'Would prepare' if dry_run else 'Prepared'} BrainMetShare: {len(records)} cases -> {dst_root}")
    print(f"Cases by split: {split_counts}")
    print(f"Labeled cases by split: {labeled_counts}")
    print(f"Existing destination case dirs: {existing_case_dirs}")
    print(f"Existing expected files: {existing_files}/{expected_files}")
    if dry_run:
        print("Dry run only: no files or manifests were written.")
    else:
        print(f"Wrote manifest: {dst_root / 'manifest.jsonl'}")
        print(f"Wrote manifest: {dst_root / 'manifest.csv'}")


def main() -> int:
    args = parse_args()
    src_root = args.src.resolve()
    dst_root = args.dst.resolve()

    records = [inspect_case(split, case_dir, dst_root) for split, case_dir in iter_cases(src_root)]
    if not records:
        raise RuntimeError(f"No cases found under {src_root / DATASET_SUBDIR}")

    missing_required = [rec for rec in records if rec.missing_files]
    if missing_required:
        examples = "; ".join(f"{rec.case_id}: {', '.join(rec.missing_files)}" for rec in missing_required[:5])
        raise FileNotFoundError(f"{len(missing_required)} cases are missing required modalities. Examples: {examples}")

    if args.dry_run:
        print_summary(records, dst_root, dry_run=True)
        return 0

    dst_exists_nonempty = dst_root.exists() and any(dst_root.iterdir())
    if dst_exists_nonempty and not (args.clean or args.overwrite or args.force_existing):
        raise FileExistsError(
            f"Destination already exists and is not empty: {dst_root}. "
            "Use --dry-run to inspect, --force-existing to add missing files/rewrite manifests, "
            "--overwrite to replace existing files, or --clean to rebuild."
        )

    if args.clean and dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for rec in records:
        prepare_case(rec, args.overwrite)
        if args.generate_png and rec.has_seg:
            generate_axial_pngs(rec, args.png_subdir, args.num_png, args.png_size, args.overwrite)

    write_manifest(records, dst_root)

    print_summary(records, dst_root, dry_run=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
