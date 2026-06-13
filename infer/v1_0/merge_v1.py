#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return re.sub(r"\s+", " ", x).strip()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def safe_float_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def safe_float_std(xs: Sequence[float]) -> float:
    return float(np.std(xs)) if xs else float("nan")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sorted_glob(pattern: str) -> List[Path]:
    return [Path(p) for p in sorted(glob.glob(pattern))]


def read_details(pattern: str) -> List[Dict[str, Any]]:
    details = []
    seen = set()
    for path in sorted_glob(pattern):
        for row in iter_jsonl(path):
            if "__row_index" not in row:
                raise ValueError(f"Missing __row_index in detail file {path}")
            idx = int(row["__row_index"])
            if idx in seen:
                raise ValueError(f"Duplicate __row_index={idx} while reading {path}")
            seen.add(idx)
            details.append(row)
    details.sort(key=lambda x: int(x["__row_index"]))
    return details


def strip_parallel_meta(detail: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(detail)
    for k in ["__row_index", "__shard_id", "__processed_index"]:
        out.pop(k, None)
    return out


def write_merged_detail(path: Path, details: List[Dict[str, Any]], keep_parallel_meta: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in details:
            out = dict(row) if keep_parallel_meta else strip_parallel_meta(row)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def group_report_rows(case_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = case_rows[0]
    return {
        "dataset": first["dataset"],
        "label_id": first["label_id"],
        "label_name": first["label_name"],
        "num_cases": len(case_rows),
        "raw_mean": safe_float_mean([r["case_raw_mean"] for r in case_rows]),
        "rewrite_mean": safe_float_mean([r["case_rewrite_mean"] for r in case_rows]),
        "raw_std": safe_float_mean([r["case_raw_std"] for r in case_rows]),
        "rewrite_std": safe_float_mean([r["case_rewrite_std"] for r in case_rows]),
        "raw_worst": safe_float_mean([r["case_raw_worst"] for r in case_rows]),
        "rewrite_worst": safe_float_mean([r["case_rewrite_worst"] for r in case_rows]),
    }


def build_report_from_details(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    case_metrics: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    for row in details:
        dataset = clean_text(row.get("dataset"))
        case_id = clean_text(row.get("case_id"))
        label_id = int(row["label_id"])
        label_name = clean_text(row.get("label_name"))
        case_key = (dataset, case_id, label_id, label_name)
        if case_key not in case_metrics:
            case_metrics[case_key] = {
                "dataset": dataset,
                "case_id": case_id,
                "label_id": label_id,
                "label_name": label_name,
                "raw_dices": [],
                "rewrite_dices": [],
            }
        case_metrics[case_key]["raw_dices"].append(float(row["raw_dice"]))
        case_metrics[case_key]["rewrite_dices"].append(float(row["rewrite_dice"]))

    grouped_case_rows: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for (_dataset, _case_id, _label_id, _label_name), val in case_metrics.items():
        grouped_case_rows[(val["dataset"], val["label_id"], val["label_name"])].append(
            {
                "dataset": val["dataset"],
                "case_id": val["case_id"],
                "label_id": val["label_id"],
                "label_name": val["label_name"],
                "case_raw_mean": safe_float_mean(val["raw_dices"]),
                "case_rewrite_mean": safe_float_mean(val["rewrite_dices"]),
                "case_raw_std": safe_float_std(val["raw_dices"]),
                "case_rewrite_std": safe_float_std(val["rewrite_dices"]),
                "case_raw_worst": float(min(val["raw_dices"])) if val["raw_dices"] else float("nan"),
                "case_rewrite_worst": float(min(val["rewrite_dices"])) if val["rewrite_dices"] else float("nan"),
            }
        )
    return [group_report_rows(cr) for _, cr in sorted(grouped_case_rows.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail_glob", required=True)
    ap.add_argument("--merged_detail", required=True)
    ap.add_argument("--merged_report", required=True)
    ap.add_argument("--keep_parallel_meta", action="store_true")
    args = ap.parse_args()

    details = read_details(args.detail_glob)
    write_merged_detail(Path(args.merged_detail), details, keep_parallel_meta=args.keep_parallel_meta)

    report_rows = build_report_from_details(details)
    write_csv(
        Path(args.merged_report),
        report_rows,
        ["dataset", "label_id", "label_name", "num_cases", "raw_mean", "rewrite_mean", "raw_std", "rewrite_std", "raw_worst", "rewrite_worst"],
    )

    print("\n=== Merged Report ===")
    for row in report_rows:
        print(f"{row['dataset']}\t{row['label_name']}\tCases:{row['num_cases']}\tRaw:{row['raw_mean']:.5f}\tRewrite:{row['rewrite_mean']:.5f}")
    print(f"Merged details: {len(details)} rows")


if __name__ == "__main__":
    main()
