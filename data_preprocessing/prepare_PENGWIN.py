#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare PENGWIN CT into the PENGWIN_CT-prepared layout.

Source layout:
    PENGWIN/PENGWIN_CT_train_images/001.mha
    PENGWIN/PENGWIN_CT_train_labels/001.mha

Prepared layout:
    PENGWIN_CT-prepared/001/001_image.nii.gz
    PENGWIN_CT-prepared/001/001_label.nii.gz
    PENGWIN_CT-prepared/001/001_label_semantic.nii.gz
    PENGWIN_CT-prepared/001/vlm_png_axial/z040.png  (optional)

Semantic remap:
    original labels  1..10 -> 1  sacrum
    original labels 11..20 -> 2  left hipbone
    original labels 21..30 -> 3  right hipbone
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


DEFAULT_SRC = Path('/home/dhm_41310/hdd/trzhang/datasets/PENGWIN')
DEFAULT_DST = Path('/home/dhm_41310/hdd/trzhang/datasets/PENGWIN_CT-prepared')
IMAGE_SUBDIR = 'PENGWIN_CT_train_images'
LABEL_SUBDIR = 'PENGWIN_CT_train_labels'
DATASET = 'PENGWIN'
PNG_GAMMA = 1.4


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_mha: Path | None
    label_mha: Path | None
    output_dir: Path

    @property
    def has_image(self) -> bool:
        return self.image_mha is not None and self.image_mha.is_file()

    @property
    def has_label(self) -> bool:
        return self.label_mha is not None and self.label_mha.is_file()

    @property
    def image_out(self) -> Path:
        return self.output_dir / f'{self.case_id}_image.nii.gz'

    @property
    def label_out(self) -> Path:
        return self.output_dir / f'{self.case_id}_label.nii.gz'

    @property
    def semantic_out(self) -> Path:
        return self.output_dir / f'{self.case_id}_label_semantic.nii.gz'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare PENGWIN CT as /PENGWIN_CT-prepared/<case>/<case>_*.nii.gz.'
    )
    parser.add_argument('--src', type=Path, default=DEFAULT_SRC, help=f'Source root. Default: {DEFAULT_SRC}')
    parser.add_argument('--dst', type=Path, default=DEFAULT_DST, help=f'Output root. Default: {DEFAULT_DST}')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing destination files.')
    parser.add_argument(
        '--force-existing',
        action='store_true',
        help='Allow writing into an existing non-empty destination without replacing existing files.',
    )
    parser.add_argument('--clean', action='store_true', help='Delete destination before preparing.')
    parser.add_argument('--dry-run', action='store_true', help='Scan and report only; do not write anything.')
    parser.add_argument('--skip-semantic', action='store_true', help='Do not create *_label_semantic.nii.gz.')
    parser.add_argument('--generate-png', action='store_true', help='Generate axial PNG previews from CT images.')
    parser.add_argument('--num-png', type=int, default=8, help='Number of PNG previews per case. Default: 8.')
    parser.add_argument('--png-size', type=int, default=448, help='Output PNG width/height. Default: 448.')
    parser.add_argument('--png-subdir', default='vlm_png_axial', help='PNG preview subdirectory name.')
    return parser.parse_args()


def natural_key(path: Path) -> tuple[object, ...]:
    text = path.stem
    parts: list[object] = []
    buf = ''
    is_digit = False
    for ch in text:
        if ch.isdigit() != is_digit and buf:
            parts.append(int(buf) if is_digit else buf.lower())
            buf = ''
        buf += ch
        is_digit = ch.isdigit()
    if buf:
        parts.append(int(buf) if is_digit else buf.lower())
    return tuple(parts)


def iter_cases(src_root: Path, dst_root: Path) -> Iterable[CaseRecord]:
    images_dir = src_root / IMAGE_SUBDIR
    labels_dir = src_root / LABEL_SUBDIR
    if not images_dir.is_dir():
        raise FileNotFoundError(f'Missing image directory: {images_dir}')
    if not labels_dir.is_dir():
        raise FileNotFoundError(f'Missing label directory: {labels_dir}')

    image_ids = {p.stem for p in images_dir.glob('*.mha') if p.is_file()}
    label_ids = {p.stem for p in labels_dir.glob('*.mha') if p.is_file()}
    for case_id in sorted(image_ids | label_ids):
        image_mha = images_dir / f'{case_id}.mha'
        label_mha = labels_dir / f'{case_id}.mha'
        yield CaseRecord(
            case_id=case_id,
            image_mha=image_mha if image_mha.is_file() else None,
            label_mha=label_mha if label_mha.is_file() else None,
            output_dir=dst_root / case_id,
        )


def assert_can_write_destination(dst_root: Path, args: argparse.Namespace) -> None:
    dst_exists_nonempty = dst_root.exists() and any(dst_root.iterdir())
    if dst_exists_nonempty and not (args.clean or args.overwrite or args.force_existing):
        raise FileExistsError(
            f'Destination already exists and is not empty: {dst_root}. '
            'Use --dry-run to inspect, --force-existing to add missing files/rewrite manifests, '
            '--overwrite to replace existing files, or --clean to rebuild.'
        )


def write_sitk_image(src: Path, dst: Path, overwrite: bool) -> None:
    import SimpleITK as sitk

    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst), useCompression=True)


def remap_label_array(seg):
    import numpy as np

    seg_new = np.zeros(seg.shape, dtype=np.uint8)
    seg_new[(seg >= 1) & (seg <= 10)] = 1
    seg_new[(seg >= 11) & (seg <= 20)] = 2
    seg_new[(seg >= 21) & (seg <= 30)] = 3
    return seg_new


def write_semantic_label(label_src: Path, dst: Path, overwrite: bool) -> None:
    import SimpleITK as sitk

    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)

    label_img = sitk.ReadImage(str(label_src))
    seg = sitk.GetArrayFromImage(label_img)
    seg_new = remap_label_array(seg)
    out_img = sitk.GetImageFromArray(seg_new)
    out_img.CopyInformation(label_img)
    sitk.WriteImage(out_img, str(dst), useCompression=True)


def normalize_ct_slice(slice_2d):
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
        raise ValueError('Empty z dimension')
    if num_png == 1:
        return [z_count // 2]
    start = int(round((z_count - 1) * 0.10))
    stop = int(round((z_count - 1) * 0.90))
    return [int(z) for z in np.linspace(start, stop, num_png).round().astype(int).tolist()]


def generate_axial_pngs(image_path: Path, png_dir: Path, num_png: int, png_size: int, overwrite: bool) -> None:
    if num_png <= 0:
        return
    if png_size <= 0:
        raise ValueError('--png-size must be positive')
    try:
        import nibabel as nib
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('--generate-png requires nibabel, numpy, and pillow') from exc

    done_marker = png_dir / f'.done_{DATASET}_{num_png}_{png_size}'
    if png_dir.exists() and overwrite:
        shutil.rmtree(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    if img.ndim > 3:
        img = np.squeeze(img)
    if img.ndim != 3:
        raise ValueError(f'Expected 3D image for {image_path}, got shape {img.shape}')

    indices = axial_slice_indices(img.shape[2], num_png)
    expected_pngs = [png_dir / f'z{z:03d}.png' for z in indices]
    if done_marker.exists() and all(p.exists() for p in expected_pngs) and not overwrite:
        return

    for old_png in png_dir.glob('*.png'):
        old_png.unlink()

    for z, out_path in zip(indices, expected_pngs):
        arr = normalize_ct_slice(img[:, :, int(z)])
        image = Image.fromarray(arr, mode='L').resize((png_size, png_size), Image.Resampling.BILINEAR)
        image.convert('RGB').save(out_path)
    done_marker.touch()


def prepare_case(rec: CaseRecord, args: argparse.Namespace) -> None:
    rec.output_dir.mkdir(parents=True, exist_ok=True)
    if rec.has_image:
        write_sitk_image(rec.image_mha, rec.image_out, args.overwrite)
    if rec.has_label:
        write_sitk_image(rec.label_mha, rec.label_out, args.overwrite)
        if not args.skip_semantic:
            write_semantic_label(rec.label_mha, rec.semantic_out, args.overwrite)
    if args.generate_png and rec.image_out.is_file():
        generate_axial_pngs(rec.image_out, rec.output_dir / args.png_subdir, args.num_png, args.png_size, args.overwrite)


def write_manifest(records: list[CaseRecord], dst_root: Path) -> None:
    with (dst_root / 'manifest.jsonl').open('w', encoding='utf-8') as f:
        for rec in records:
            obj = {
                'dataset': DATASET,
                'case_id': rec.case_id,
                'case_dir': str(rec.output_dir),
                'img_path': str(rec.image_out) if rec.has_image else '',
                'label_path': str(rec.label_out) if rec.has_label else '',
                'semantic_label_path': str(rec.semantic_out) if rec.has_label else '',
                'has_image': rec.has_image,
                'has_label': rec.has_label,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    with (dst_root / 'manifest.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ('dataset', 'case_id', 'case_dir', 'img_path', 'label_path', 'semantic_label_path', 'has_image', 'has_label')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    'dataset': DATASET,
                    'case_id': rec.case_id,
                    'case_dir': rec.output_dir,
                    'img_path': rec.image_out if rec.has_image else '',
                    'label_path': rec.label_out if rec.has_label else '',
                    'semantic_label_path': rec.semantic_out if rec.has_label else '',
                    'has_image': rec.has_image,
                    'has_label': rec.has_label,
                }
            )


def print_summary(records: list[CaseRecord], dst_root: Path, dry_run: bool) -> None:
    image_count = sum(1 for rec in records if rec.has_image)
    label_count = sum(1 for rec in records if rec.has_label)
    existing_case_dirs = sum(1 for rec in records if rec.output_dir.exists())
    existing_expected = 0
    expected = 0
    for rec in records:
        paths: list[Path] = []
        if rec.has_image:
            paths.append(rec.image_out)
        if rec.has_label:
            paths.extend([rec.label_out, rec.semantic_out])
        expected += len(paths)
        existing_expected += sum(1 for p in paths if p.exists())

    print(f"{'Would prepare' if dry_run else 'Prepared'} PENGWIN: {len(records)} cases -> {dst_root}")
    print(f'Cases with images: {image_count}')
    print(f'Cases with labels: {label_count}')
    print(f'Existing destination case dirs: {existing_case_dirs}')
    print(f'Existing expected files: {existing_expected}/{expected}')
    if dry_run:
        print('Dry run only: no files or manifests were written.')
    else:
        print(f"Wrote manifest: {dst_root / 'manifest.jsonl'}")
        print(f"Wrote manifest: {dst_root / 'manifest.csv'}")


def main() -> int:
    args = parse_args()
    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    records = list(iter_cases(src_root, dst_root))
    if not records:
        raise RuntimeError(f'No cases found under {src_root}')

    if args.dry_run:
        print_summary(records, dst_root, dry_run=True)
        return 0

    assert_can_write_destination(dst_root, args)
    if args.clean and dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for rec in records:
        prepare_case(rec, args)

    write_manifest(records, dst_root)
    print_summary(records, dst_root, dry_run=False)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
