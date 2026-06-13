#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

NUM_RE = re.compile(r"(\d+)")
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-/]+")
STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "and", "or", "on", "in", "with", "this", "that", "please",
    "segment", "delineate", "outline", "mark", "identify", "locate", "need", "region", "structure", "organ",
    "segmentation", "show", "shows", "image", "images", "case", "view", "scan", "provided", "review",
}


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return re.sub(r"\s+", " ", x).strip()


def tokenize(text: str) -> Set[str]:
    toks = TOKEN_RE.findall(clean_text(text).lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


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


@dataclass
class SkillArtifact:
    skill_id: str
    source_key: str
    tag: str
    content: str
    audit: str
    dataset: str
    style_bucket: str
    canonical: str
    aliases: List[str]
    vis_cue: str
    score_hint: float
    dice_gain: float = 0.0


class SkillBank:
    def __init__(self):
        self.skills: Dict[str, SkillArtifact] = {}
        self.feature_cache: Dict[str, Dict[str, Any]] = {}

    def _rebuild_cache(self):
        self.feature_cache.clear()
        for skill in self.skills.values():
            self.feature_cache[skill.skill_id] = {
                "aliases_lc": {a.lower() for a in skill.aliases},
                "tokens": tokenize(skill.content + " " + skill.tag),
            }

    def add(self, skill: SkillArtifact) -> bool:
        if skill.skill_id in self.skills:
            return False
        self.skills[skill.skill_id] = skill
        self.feature_cache[skill.skill_id] = {
            "aliases_lc": {a.lower() for a in skill.aliases},
            "tokens": tokenize(skill.content + " " + skill.tag),
        }
        return True

    def remove(self, skill_id: str) -> bool:
        if skill_id in self.skills:
            del self.skills[skill_id]
            if skill_id in self.feature_cache:
                del self.feature_cache[skill_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self.skills)

    def deduplicate_and_fuse(self, max_per_group: int = 5) -> int:
        groups = defaultdict(list)
        for skill in self.skills.values():
            key = (skill.dataset, skill.tag, skill.canonical.lower())
            groups[key].append(skill)

        new_skills = {}
        fused_count = 0

        for key, group in groups.items():
            if len(group) <= max_per_group:
                for s in group:
                    new_skills[s.skill_id] = s
                continue

            # Keep v5 semantics: sort only by dice_gain, relying on insertion order for ties.
            group.sort(key=lambda s: s.dice_gain, reverse=True)

            kept_skills = group[:max_per_group]
            dropped_skills = group[max_per_group:]

            best_skill = kept_skills[0]
            merged_aliases = set(best_skill.aliases)
            for drop_s in dropped_skills:
                merged_aliases.update(drop_s.aliases)
                fused_count += 1

            best_skill.aliases = list(merged_aliases)

            for s in kept_skills:
                new_skills[s.skill_id] = s

        self.skills = new_skills
        self._rebuild_cache()
        return fused_count

    def cull(self, max_size: int) -> int:
        if len(self.skills) <= max_size:
            return 0
        # Keep v5 semantics: sort only by dice_gain, relying on insertion order for ties.
        sorted_skills = sorted(self.skills.values(), key=lambda s: s.dice_gain, reverse=True)
        culled_count = len(self.skills) - max_size
        self.skills = {s.skill_id: s for s in sorted_skills[:max_size]}
        self._rebuild_cache()
        return culled_count

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for skill in sorted(self.skills.values(), key=lambda s: (s.dataset, s.tag, s.skill_id)):
                row = skill_to_row(skill)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "SkillBank":
        bank = cls()
        if not path.exists():
            return bank
        for row in iter_jsonl(path):
            bank.add(skill_from_row(row))
        return bank


def skill_from_row(row: Dict[str, Any]) -> SkillArtifact:
    return SkillArtifact(
        skill_id=clean_text(row.get("skill_id")),
        source_key=clean_text(row.get("source_key")),
        tag=clean_text(row.get("tag")),
        content=clean_text(row.get("content")),
        audit=clean_text(row.get("audit")),
        dataset=clean_text(row.get("dataset")),
        style_bucket=clean_text(row.get("style_bucket")),
        canonical=clean_text(row.get("canonical")),
        aliases=list(row.get("aliases") or []),
        vis_cue=clean_text(row.get("vis_cue")),
        score_hint=float(row.get("score_hint", 0.0) or 0.0),
        dice_gain=float(row.get("dice_gain", row.get("score_hint", 0.0)) or 0.0),
    )


def skill_to_row(skill: SkillArtifact) -> Dict[str, Any]:
    return {
        "skill_id": skill.skill_id,
        "source_key": skill.source_key,
        "tag": skill.tag,
        "content": skill.content,
        "audit": skill.audit,
        "dataset": skill.dataset,
        "style_bucket": skill.style_bucket,
        "canonical": skill.canonical,
        "aliases": skill.aliases,
        "vis_cue": skill.vis_cue,
        "score_hint": skill.score_hint,
        "dice_gain": skill.dice_gain,
    }


def safe_float_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def safe_float_std(xs: Sequence[float]) -> float:
    return float(np.std(xs)) if xs else float("nan")


def finite_float_or_nan(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def build_latency_summary_from_details(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute latency using sums across all valid cases, not mean of per-case ratios."""
    raw_times = []
    rewrite_times = []
    vlm_times = []
    ft_seg_times = []
    skipped_missing_timing = 0

    for row in details:
        raw_t = finite_float_or_nan(row.get("raw_seg_time_sec"))
        ft_total_t = finite_float_or_nan(row.get("ft_total_time_sec"))
        vlm_t = finite_float_or_nan(row.get("vlm_generate_time_sec"))
        ft_seg_t = finite_float_or_nan(row.get("ft_seg_time_sec"))
        if raw_t > 0 and ft_total_t >= 0:
            raw_times.append(raw_t)
            rewrite_times.append(ft_total_t)
            if math.isfinite(vlm_t):
                vlm_times.append(vlm_t)
            if math.isfinite(ft_seg_t):
                ft_seg_times.append(ft_seg_t)
        else:
            skipped_missing_timing += 1

    sum_raw = float(sum(raw_times))
    sum_rewrite = float(sum(rewrite_times))
    n = len(raw_times)
    ratio_pct = (sum_rewrite / sum_raw * 100.0) if sum_raw > 0 else float("nan")
    overhead_pct = ratio_pct - 100.0 if math.isfinite(ratio_pct) else float("nan")

    return [{
        "scope": "ALL",
        "num_cases_with_timing": n,
        "num_cases_total": len(details),
        "num_cases_skipped_missing_timing": skipped_missing_timing,
        "raw_total_time_sec": sum_raw,
        "rewrite_total_time_sec": sum_rewrite,
        "raw_mean_time_sec": (sum_raw / n) if n else float("nan"),
        "rewrite_mean_time_sec": (sum_rewrite / n) if n else float("nan"),
        "vlm_generate_mean_time_sec": (float(sum(vlm_times)) / len(vlm_times)) if vlm_times else float("nan"),
        "ft_seg_mean_time_sec": (float(sum(ft_seg_times)) / len(ft_seg_times)) if ft_seg_times else float("nan"),
        "latency_ratio_vs_raw_pct": ratio_pct,
        "latency_overhead_vs_raw_pct": overhead_pct,
    }]


def group_report_rows(case_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = case_rows[0]
    return {
        "dataset": first["dataset"],
        "label_id": first["label_id"],
        "label_name": first["label_name"],
        "num_cases": len(case_rows),
        "raw_mean": safe_float_mean([r["case_raw_mean"] for r in case_rows]),
        "ft_mean": safe_float_mean([r["case_ft_mean"] for r in case_rows]),
        "raw_std": safe_float_mean([r["case_raw_std"] for r in case_rows]),
        "ft_std": safe_float_mean([r["case_ft_std"] for r in case_rows]),
        "raw_worst": safe_float_mean([r["case_raw_worst"] for r in case_rows]),
        "ft_worst": safe_float_mean([r["case_ft_worst"] for r in case_rows]),
    }


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


def read_new_skill_records(pattern: str) -> List[Dict[str, Any]]:
    records = []
    for path in sorted_glob(pattern):
        for row in iter_jsonl(path):
            if "row_index" not in row or "skill" not in row:
                raise ValueError(f"Invalid new skill record in {path}: expected row_index and skill")
            records.append(row)
    records.sort(key=lambda x: int(x["row_index"]))
    return records


def read_audit_records(pattern: str) -> List[Dict[str, Any]]:
    records = []
    for path in sorted_glob(pattern):
        for row in iter_jsonl(path):
            if "row_index" not in row or "skill_id" not in row:
                raise ValueError(f"Invalid audit record in {path}: expected row_index and skill_id")
            records.append(row)
    records.sort(key=lambda x: (int(x["row_index"]), clean_text(x.get("skill_id"))))
    return records


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
                "dataset": dataset, "case_id": case_id, "label_id": label_id, "label_name": label_name,
                "raw_dices": [], "ft_dices": [],
            }
        case_metrics[case_key]["raw_dices"].append(float(row["raw_dice"]))
        case_metrics[case_key]["ft_dices"].append(float(row["ft_dice"]))

    grouped_case_rows: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for (_dataset, _case_id, _label_id, _label_name), val in case_metrics.items():
        grouped_case_rows[(val["dataset"], val["label_id"], val["label_name"])].append(
            {
                "dataset": val["dataset"], "case_id": val["case_id"], "label_id": val["label_id"], "label_name": val["label_name"],
                "case_raw_mean": safe_float_mean(val["raw_dices"]), "case_ft_mean": safe_float_mean(val["ft_dices"]),
                "case_raw_std": safe_float_std(val["raw_dices"]), "case_ft_std": safe_float_std(val["ft_dices"]),
                "case_raw_worst": float(min(val["raw_dices"])) if val["raw_dices"] else float("nan"),
                "case_ft_worst": float(min(val["ft_dices"])) if val["ft_dices"] else float("nan"),
            }
        )
    return [group_report_rows(cr) for _, cr in sorted(grouped_case_rows.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", choices=["round0_no_skill", "roundN_with_skill"], required=True)
    ap.add_argument("--round_name", required=True)
    ap.add_argument("--skill_bank_dir", required=True)
    ap.add_argument("--skill_bank_in", default="")
    ap.add_argument("--detail_glob", required=True)
    ap.add_argument("--new_skill_glob", required=True)
    ap.add_argument("--audit_glob", required=True)
    ap.add_argument("--merged_detail", required=True)
    ap.add_argument("--merged_report", required=True)
    ap.add_argument("--latency_summary_out", default="", help="Optional CSV path for all-case latency summary. Defaults to <merged_report>_latency.csv")
    ap.add_argument("--merged_skill_latest", required=True)
    ap.add_argument("--merged_skill_round", required=True)
    ap.add_argument("--skill_bank_max_ratio", type=float, default=0.05)
    ap.add_argument("--skill_max_per_group", type=int, default=10)
    ap.add_argument("--audit_risk_penalty", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep_parallel_meta", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    rows = list(iter_jsonl(input_path))
    if args.limit > 0:
        rows = rows[:args.limit]
    max_skill_bank_size = max(1, int(len(rows) * args.skill_bank_max_ratio))

    if args.mode == "round0_no_skill":
        skill_bank = SkillBank()
        print("[Merge Mode] round0_no_skill: start from empty skill bank")
    else:
        if args.skill_bank_in:
            bank_path = Path(args.skill_bank_in)
        else:
            bank_path = Path(args.skill_bank_dir) / "latest.jsonl"
        skill_bank = SkillBank.load(bank_path)
        print(f"[Merge Mode] roundN_with_skill: loaded {len(skill_bank)} skills from {bank_path}")

    details = read_details(args.detail_glob)
    write_merged_detail(Path(args.merged_detail), details, keep_parallel_meta=args.keep_parallel_meta)
    report_rows = build_report_from_details(details)
    write_csv(Path(args.merged_report), report_rows, ["dataset", "label_id", "label_name", "num_cases", "raw_mean", "ft_mean", "raw_std", "ft_std", "raw_worst", "ft_worst"])

    latency_summary_path = Path(args.latency_summary_out) if args.latency_summary_out else Path(args.merged_report).with_name(Path(args.merged_report).stem + "_latency.csv")
    latency_summary_rows = build_latency_summary_from_details(details)
    latency_fieldnames = [
        "scope",
        "num_cases_with_timing",
        "num_cases_total",
        "num_cases_skipped_missing_timing",
        "raw_total_time_sec",
        "rewrite_total_time_sec",
        "raw_mean_time_sec",
        "rewrite_mean_time_sec",
        "vlm_generate_mean_time_sec",
        "ft_seg_mean_time_sec",
        "latency_ratio_vs_raw_pct",
        "latency_overhead_vs_raw_pct",
    ]
    write_csv(latency_summary_path, latency_summary_rows, latency_fieldnames)

    new_skill_records = read_new_skill_records(args.new_skill_glob)
    added_skills = 0
    duplicate_skills = 0
    for rec in new_skill_records:
        skill = skill_from_row(rec["skill"])
        if skill_bank.add(skill):
            added_skills += 1
        else:
            duplicate_skills += 1

    audit_records = read_audit_records(args.audit_glob)
    skill_audit_stats: Dict[str, Dict[str, float]] = {}
    for rec in audit_records:
        sid = clean_text(rec.get("skill_id"))
        if sid not in skill_audit_stats:
            skill_audit_stats[sid] = {"uses": 0.0, "pos_gain": 0.0, "neg_drop": 0.0}
        skill_audit_stats[sid]["uses"] += float(rec.get("uses", 1) or 1)
        skill_audit_stats[sid]["pos_gain"] += float(rec.get("pos_gain", 0.0) or 0.0)
        skill_audit_stats[sid]["neg_drop"] += float(rec.get("neg_drop", 0.0) or 0.0)

    audited_count = 0
    purged_count = 0
    print("\n" + "="*30 + " V5 PARALLEL MERGE AUDIT TRIBUNAL " + "="*30)
    for sid, stats in skill_audit_stats.items():
        if stats["uses"] > 0:
            audited_count += 1
            utility = stats["pos_gain"] - (args.audit_risk_penalty * stats["neg_drop"])
            if utility < 0:
                if skill_bank.remove(sid):
                    purged_count += 1
                    print(f" [PURGE] {sid[:12]}... | Util: {utility:.3f} (Pos: +{stats['pos_gain']:.3f}, Neg: -{stats['neg_drop']:.3f} x {args.audit_risk_penalty})")
            else:
                if sid in skill_bank.skills:
                    skill_bank.skills[sid].dice_gain = utility / stats["uses"]
    print(f"Audit Complete: Checked {audited_count} active skills, purged {purged_count} toxic skills.")
    print("="*100 + "\n")

    fused_count = skill_bank.deduplicate_and_fuse(max_per_group=args.skill_max_per_group)
    culled_count = skill_bank.cull(max_size=max_skill_bank_size)

    latest_path = Path(args.merged_skill_latest)
    round_path = Path(args.merged_skill_round)
    skill_bank.save(latest_path)
    skill_bank.save(round_path)

    print("\n=== Merged Report ===")
    for row in report_rows:
        print(f"{row['dataset']}\t{row['label_name']}\tCases:{row['num_cases']}\tRaw:{row['raw_mean']:.5f}\tFT:{row['ft_mean']:.5f}")
    if latency_summary_rows:
        lat = latency_summary_rows[0]
        print(
            f"\nLatency Summary -> Cases: {lat['num_cases_with_timing']}/{lat['num_cases_total']} | "
            f"Raw total: {lat['raw_total_time_sec']:.3f}s | Rewrite+Seg total: {lat['rewrite_total_time_sec']:.3f}s | "
            f"Ratio: {lat['latency_ratio_vs_raw_pct']:.2f}% | Overhead: {lat['latency_overhead_vs_raw_pct']:.2f}%"
        )
        print(f"Latency summary CSV: {latency_summary_path}")

    print(
        f"\nFinal Bank Stats -> Replay Add: {added_skills} | Duplicate Candidates: {duplicate_skills} | "
        f"Audit Purge: {purged_count} | Fused: {fused_count} | Culled: {culled_count}"
    )
    print(f"Final skill bank size: {len(skill_bank)}")
    print(f"Merged details: {len(details)} rows")


if __name__ == "__main__":
    main()
