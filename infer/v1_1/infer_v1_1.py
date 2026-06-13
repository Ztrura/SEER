#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from peft import PeftModel

DEFAULT_SYSTEM_PROMPT = (
    "You are a medical image prompt-normalization assistant for text-guided 3D segmentation. "
    "Given multi-slice images and a raw segmentation request, produce three parts in this exact format: "
    "<evidence>...</evidence><rationale>...</rationale><answer>...</answer>. "
    "In <evidence>, describe only objective visual observations and imaging context. "
    "In <rationale>, explain briefly how the raw request is normalized based on the evidence. "
    "If a <skill_bank> is provided, explicitly choose the single most relevant skill and mention that choice in <rationale>. "
    "In <answer>, provide the final normalized segmentation prompt."
)

DEFAULT_USER_PROMPT_NO_SKILL = (
    "Review the provided medical image slices and the raw segmentation request. "
    "Return ONLY three parts (objective evidence, a short rationale, and the final normalized reprompt) in this exact format:"
    "\n<evidence>...</evidence>\n<rationale>...</rationale>\n<answer>...</answer>"
)
DEFAULT_USER_PROMPT_WITH_SKILL = (
    "Review the provided medical image slices, the raw segmentation request, and the candidate skill bank. "
    "Choose the single most relevant skill, state the choice in the rationale."
    "Then return ONLY three parts (objective evidence, a short rationale, and the final normalized reprompt) in this exact format:"
    "\n<evidence>...</evidence>\n<rationale>...</rationale>\n<answer>...</answer>"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
RATIONALE_RE = re.compile(r"<rationale>(.*?)</rationale>", re.IGNORECASE | re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<evidence>.*?</evidence>\s*<rationale>.*?</rationale>\s*<answer>.*?</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
NUM_RE = re.compile(r"(\d+)")
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-/]+")
STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "and", "or", "on", "in", "with", "this", "that", "please",
    "segment", "delineate", "outline", "mark", "identify", "locate", "need", "region", "structure", "organ",
    "segmentation", "show", "shows", "image", "images", "case", "view", "scan", "provided", "review",
}
STYLE_TO_SKILL_TAG = {
    "ambiguous_abbreviation": "ABBR_DISAMBIGUATE",
    "messy_or_misspelled": "MESSY_INPUT_REPAIR",
    "out_of_region_reference": "OUT_OF_REGION_REJECT",
    "clinical_request": "CLINICAL_REQUEST_NORMALIZE",
    "report_style": "REPORT_STYLE_PARSE",
    "verbose_request": "VERBOSE_PRUNE",
    "formal_task": "PROMPT_NORMALIZE",
    "abbr_or_short": "ABBR_DISAMBIGUATE",
}
DATASET_VIS_CUE = {
    "HVSMR": "CARDIAC_CHAMBERS",
    "brats2021": "BRAIN_LESION",
    "brats2024": "BRAIN_LESION",
    "isles2022": "BRAIN_LESION",
    "BrainMetShare": "BRAIN_LESION",
    "FLARE": "ABDOMINAL_ORGANS",
    "PENGWIN": "PELVIC_BONES",
}


def natural_key(s: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in NUM_RE.split(s)]


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return re.sub(r"\s+", " ", x).strip()


def tokenize(text: str) -> Set[str]:
    toks = TOKEN_RE.findall(clean_text(text).lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def collect_pngs(case_dir: Path, png_subdir: str, max_images: int) -> List[Path]:
    png_dir = case_dir / png_subdir
    if not png_dir.exists() or not png_dir.is_dir():
        return []
    files = [p for p in png_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files = sorted(files, key=lambda p: natural_key(p.name))
    if max_images > 0:
        files = files[:max_images]
    return files


def parse_answer_text(generated_text: str) -> str:
    text = clean_text(generated_text)
    m = ANSWER_RE.search(text)
    if m:
        return clean_text(m.group(1))
    text = re.sub(r"<[^>]+>", " ", text)
    return clean_text(text)


def has_valid_format(generated_text: str) -> bool:
    return bool(FORMAT_RE.match(clean_text(generated_text)))


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    p = int(pred.sum())
    g = int(gt.sum())
    if p == 0 and g == 0:
        return 1.0
    if p == 0 or g == 0:
        return 0.0
    inter = int((pred & gt).sum())
    return float((2.0 * inter) / (p + g))


def sync_cuda_if_available() -> None:
    """Synchronize CUDA kernels before/after timing so GPU latency is not under-counted."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def finite_float_or_nan(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def parse_raw_cache_entry(entry: Any) -> Tuple[Optional[float], float, bool]:
    """Return (raw_dice, raw_seg_time_sec, legacy_format).

    Old caches stored only a float raw_dice. New caches store raw_dice and the
    raw-prompt segmentation latency needed for fair round1/round2 latency accounting.
    """
    if isinstance(entry, dict):
        raw_dice = entry.get("raw_dice")
        if raw_dice is None:
            return None, float("nan"), False
        raw_time = entry.get("raw_seg_time_sec", entry.get("raw_time_sec"))
        return float(raw_dice), finite_float_or_nan(raw_time), False
    if entry is not None:
        return float(entry), float("nan"), True
    return None, float("nan"), False


def make_raw_cache_entry(raw_dice: float, raw_seg_time_sec: float) -> Dict[str, float]:
    return {
        "raw_dice": float(raw_dice),
        "raw_seg_time_sec": float(raw_seg_time_sec),
    }


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

    def render_block(self, rank: int) -> str:
        return (
            f"[{rank}]\n"
            f"Tag: {self.tag}\n"
            f"Content: {self.content}\n"
            f"Audit: {self.audit}"
        )


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
        """V5: Remove a toxic skill during the Audit phase."""
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
        sorted_skills = sorted(self.skills.values(), key=lambda s: s.dice_gain, reverse=True)
        culled_count = len(self.skills) - max_size
        self.skills = {s.skill_id: s for s in sorted_skills[:max_size]}
        self._rebuild_cache()
        return culled_count

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for skill in sorted(self.skills.values(), key=lambda s: (s.dataset, s.tag, s.skill_id)):
                row = {
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
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "SkillBank":
        bank = cls()
        if not path.exists():
            return bank
        for row in iter_jsonl(path):
            skill = SkillArtifact(
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
            bank.add(skill)
        return bank

    def retrieve_topk_scored(self, row: Dict[str, Any], k: int) -> List[Tuple[SkillArtifact, float]]:
        if not self.skills or k <= 0:
            return []
        row_dataset = clean_text(row.get("dataset"))
        row_style = clean_text(row.get("style_bucket"))
        row_canonical = clean_text(row.get("label_desc") or row.get("label_name")).lower()
        row_aliases = {
            clean_text(row.get("label_name")).lower(),
            clean_text(row.get("label_desc")).lower(),
            row_canonical,
        }
        row_aliases.discard("")
        row_tokens = tokenize(clean_text(row.get("raw_prompt")))
        row_vis = DATASET_VIS_CUE.get(row_dataset, "GENERIC_MEDICAL_IMAGE")
        row_key = build_source_key(row)

        scored: List[Tuple[float, SkillArtifact]] = []
        for skill in self.skills.values():
            if skill.source_key == row_key:
                continue
            sf = self.feature_cache[skill.skill_id]
            score = 0.0
            if skill.dataset == row_dataset:
                score += 3.0
            if skill.style_bucket == row_style:
                score += 4.0
            if skill.canonical.lower() == row_canonical:
                score += 6.0
            if row_aliases & sf["aliases_lc"]:
                score += 3.0
            if skill.vis_cue == row_vis:
                score += 2.0
            score += 0.4 * len(row_tokens & sf["tokens"])
            score += 0.2 * skill.score_hint
            scored.append((score, skill))
        scored.sort(key=lambda x: (-x[0], x[1].skill_id))
        return [(skill, score) for score, skill in scored[:k]]


def normalize_aliases(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for a in [row.get("label_name"), row.get("label_desc")]:
        aa = clean_text(a)
        if aa and aa.lower() not in seen:
            out.append(aa)
            seen.add(aa.lower())
    return out


def choose_short_alias(aliases: Sequence[str], canonical: str) -> str:
    cset = canonical.lower().replace("_", " ")
    candidates: List[Tuple[int, int, str]] = []
    for a in aliases:
        aa = clean_text(a)
        if not aa:
            continue
        if aa.lower().replace("_", " ") == cset:
            continue
        score = int(len(aa) <= 6) + int(aa.isupper()) + int(" " not in aa)
        candidates.append((score, len(aa), aa))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1], x[2].lower()))
        return candidates[0][2]
    return clean_text(canonical)


def build_source_key(row: Dict[str, Any]) -> str:
    return "|".join([
        clean_text(row.get("dataset")),
        clean_text(row.get("case_id")),
        clean_text(row.get("label_id")),
        clean_text(row.get("raw_prompt")),
    ])


def build_skill_from_pair(
    row: Dict[str, Any],
    generated_text: str,
    reprompt: str,
    ft_dice: float,
    skill_gain: float,
) -> SkillArtifact:
    dataset = clean_text(row.get("dataset"))
    style_bucket = clean_text(row.get("style_bucket"))
    tag = STYLE_TO_SKILL_TAG.get(style_bucket, "PROMPT_NORMALIZE")
    canonical = clean_text(row.get("label_desc") or row.get("label_name") or reprompt)
    aliases = normalize_aliases(row)
    short_alias = choose_short_alias(aliases, canonical)
    vis_cue = DATASET_VIS_CUE.get(dataset, "GENERIC_MEDICAL_IMAGE")

    if tag == "ABBR_DISAMBIGUATE":
        content = (
            f"{short_alias}->{reprompt}; "
            f"trig={{RAW:ABBR({short_alias}), VIS:{vis_cue}, DS:{dataset}}}; "
            f"act={{EXPAND_TO_CANONICAL, KEEP_MINIMAL}}"
        )
    elif tag == "MESSY_INPUT_REPAIR":
        content = (
            f"target={reprompt}; trig={{RAW:MESSY_OR_MISSPELLED, VIS:{vis_cue}, DS:{dataset}}}; "
            f"act={{REPAIR_SURFACE_FORM, PRESERVE_TARGET, KEEP_MINIMAL}}"
        )
    elif tag == "OUT_OF_REGION_REJECT":
        content = (
            f"target={reprompt}; trig={{RAW:CROSS_REGION_TERM, VIS:{vis_cue}, DS:{dataset}}}; "
            f"act={{DROP_OUT_OF_REGION_DETAIL, PRESERVE_IN_REGION_TARGET}}"
        )
    elif tag == "CLINICAL_REQUEST_NORMALIZE":
        content = (
            f"target={reprompt}; trig={{RAW:QUICK_CLINICAL_REQUEST, VIS:{vis_cue}, DS:{dataset}}}; "
            f"act={{NORMALIZE_TO_SEGMENT_COMMAND, KEEP_MINIMAL}}"
        )
    elif tag == "REPORT_STYLE_PARSE":
        content = (
            f"target={reprompt}; trig={{RAW:REPORT_STYLE, VIS:{vis_cue}, DS:{dataset}}}; "
            f"act={{PARSE_REPORT_LANGUAGE, CONVERT_TO_SEGMENT_COMMAND}}"
        )
    elif tag == "VERBOSE_PRUNE":
        content = (
            f"target={reprompt}; trig={{RAW:VERBOSE_REQUEST, VIS:{vis_cue}, DS:{dataset}}}; "
            f"act={{REMOVE_UNSUPPORTED_CONTEXT, KEEP_EFFECTIVE_CONSTRAINTS, KEEP_MINIMAL}}"
        )
    else:
        content = f"target={reprompt}; trig={{VIS:{vis_cue}, DS:{dataset}}}; act={{NORMALIZE_PROMPT}}"

    source_key = build_source_key(row)
    skill_id = f"seer:{stable_hash(source_key + '|' + content)}"
    audit = (
        f"id={skill_id}; src={dataset}/{clean_text(row.get('case_id'))}/L{clean_text(row.get('label_id'))}; "
        f"style={style_bucket}; reward={{ft_dice:{ft_dice:.4f}, gain:{skill_gain:.4f}}}"
    )
    return SkillArtifact(
        skill_id=skill_id,
        source_key=source_key,
        tag=tag,
        content=content,
        audit=audit,
        dataset=dataset,
        style_bucket=style_bucket,
        canonical=canonical,
        aliases=aliases,
        vis_cue=vis_cue,
        score_hint=float(ft_dice),
        dice_gain=float(skill_gain),
    )


class VLMRepromptModel:
    def __init__(
        self,
        model_path: str,
        processor_path: Optional[str] = None,
        adapters: str = "",
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        attn_impl: str = "flash_attention_2",
        max_new_tokens: int = 256,
    ):
        self.model_path = model_path
        self.processor_path = processor_path or model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = "auto"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map={"": device} if device != "auto" else "auto",
            attn_implementation=attn_impl,
        )
        if adapters:
            self.model = PeftModel.from_pretrained(self.model, adapters)
        self.processor = AutoProcessor.from_pretrained(self.processor_path)
        self.model.eval()

    @torch.inference_mode()
    def generate_full_response(
        self,
        image_paths: Sequence[Path],
        raw_prompt: str,
        system_prompt: str,
        user_prompt: str,
        skill_bank_text: str = "",
    ) -> str:
        content: List[Dict[str, Any]] = []
        for p in image_paths:
            content.append({"type": "image", "image": str(p)})
        text = user_prompt
        if skill_bank_text:
            text += f"\n\n<skill_bank>\n{skill_bank_text}\n</skill_bank>"
        text += f"\n\n<raw_prompt>{raw_prompt}</raw_prompt>"
        content.append({"type": "text", "text": text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return clean_text(output_text[0] if output_text else "")


class VoxTellEvaluator:
    def __init__(self, model_dir: str, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.predictor = VoxTellPredictor(model_dir=model_dir, device=self.device)
        self.io = NibabelIOWithReorient()
        self._img_cache: Dict[str, Tuple[np.ndarray, Any]] = {}
        self._seg_cache: Dict[str, np.ndarray] = {}

    def load_image(self, img_path: str) -> Tuple[np.ndarray, Any]:
        if img_path not in self._img_cache:
            img, props = self.io.read_images([img_path])
            if isinstance(props, (list, tuple)) and len(props) == 1:
                props = props[0]
            self._img_cache[img_path] = (img, props)
        return self._img_cache[img_path]

    def load_seg(self, seg_path: str) -> np.ndarray:
        if seg_path not in self._seg_cache:
            seg_img, _ = self.io.read_images([seg_path])
            seg = seg_img[0] if (seg_img.ndim == 4 and seg_img.shape[0] == 1) else seg_img
            self._seg_cache[seg_path] = np.asarray(seg)
        return self._seg_cache[seg_path]

    def predict_mask(self, img_path: str, prompt: str) -> np.ndarray:
        img, _props = self.load_image(img_path)
        seg = self.predictor.predict_single_image(img, [prompt]).astype(np.uint8)
        if seg.ndim == 4:
            return seg[0]
        return seg

    def evaluate_dice(self, img_path: str, seg_path: str, label_id: int, prompt: str) -> float:
        pred = self.predict_mask(img_path, prompt)
        gt = (self.load_seg(seg_path) == int(label_id)).astype(np.uint8)
        return dice_binary(pred, gt)

    def evaluate_dice_timed(self, img_path: str, seg_path: str, label_id: int, prompt: str) -> Tuple[float, float]:
        """Return (dice, segmentation_latency_sec).

        Timing covers the VoxTell prediction call only. Image/GT loading and Dice
        computation are intentionally outside the timed window so raw and rewritten
        prompts are compared on segmentation inference latency rather than I/O/cache effects.
        """
        img, _props = self.load_image(img_path)
        gt = (self.load_seg(seg_path) == int(label_id)).astype(np.uint8)

        sync_cuda_if_available()
        t0 = time.perf_counter()
        pred = self.predictor.predict_single_image(img, [prompt]).astype(np.uint8)
        if pred.ndim == 4:
            pred = pred[0]
        sync_cuda_if_available()
        seg_time_sec = time.perf_counter() - t0

        return dice_binary(pred, gt), float(seg_time_sec)


def safe_float_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def safe_float_std(xs: Sequence[float]) -> float:
    return float(np.std(xs)) if xs else float("nan")


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


def set_reproducible_seed(seed: int, deterministic_torch: bool = False) -> None:
    # Local imports keep the rest of the original v5_frozen code unchanged.
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Test JSONL with raw_prompt/img_path/seg_path/label_id, etc.")
    ap.add_argument("--model", required=True, help="Fine-tuned Qwen3-VL model/checkpoint path.")
    ap.add_argument("--adapters", default="", help="Path to LoRA adapter directory.")
    ap.add_argument("--processor_path", default="", help="Optional processor path; default uses --model.")
    ap.add_argument("--voxtell_model_dir", default="/home/dhm_41310/hdd/trzhang/models/VoxTell/voxtell_v1.1", help="VoxTell model directory.")
    ap.add_argument("--vlm_device", default="cuda:0")
    ap.add_argument("--seg_device", default="cuda:0")
    ap.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    ap.add_argument("--attn_impl", default="flash_attention_2")
    ap.add_argument("--png_subdir", default="vlm_png_axial")
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--strict_num_images", action="store_true")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--user_prompt_no_skill", default=DEFAULT_USER_PROMPT_NO_SKILL)
    ap.add_argument("--user_prompt_with_skill", default=DEFAULT_USER_PROMPT_WITH_SKILL)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=["round0_no_skill", "roundN_with_skill"], required=True)
    ap.add_argument("--round_name", default="round0")
    ap.add_argument("--detail_out", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--skill_bank_dir", required=True)
    ap.add_argument("--skill_bank_in", default="", help="Optional existing skill bank jsonl. If empty in roundN_with_skill, load skill_bank_dir/latest.jsonl")
    ap.add_argument("--skill_bank_topk", type=int, default=3)
    ap.add_argument("--skill_use_min_score", type=float, default=0.0, help="Only use retrieved skills whose relevance score >= this threshold.")
    ap.add_argument("--skill_min_gain", type=float, default=0.0)
    ap.add_argument("--skill_min_ft_dice", type=float, default=0.5)
    ap.add_argument("--skill_require_format", action="store_true")

    # Kept for CLI compatibility; pruning/final bank saving is done by merge script.
    ap.add_argument("--skill_bank_max_ratio", type=float, default=0.05, help="Max skill bank size as ratio of test set size (handled by merge script in parallel mode)")
    ap.add_argument("--skill_max_per_group", type=int, default=10, help="Max variants per group (handled by merge script in parallel mode)")
    ap.add_argument("--raw_dice_cache", default="", help="Path to per-shard cache file to read/write raw prompt dice scores and raw segmentation latency")
    ap.add_argument("--recompute_missing_raw_time", action="store_true", help="Re-run raw prompt segmentation if an existing cache lacks raw_seg_time_sec (legacy cache support).")
    ap.add_argument("--audit_risk_penalty", type=float, default=2.0, help="Multiplier for drops during audit; merge script applies global audit")
    ap.add_argument("--freeze_skill_bank_during_round", action="store_true", help="Use only the skill bank loaded at round start for retrieval; newly harvested skills become available next round.")

    # Parallel worker controls.
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of parallel shards.")
    ap.add_argument("--shard_id", type=int, default=0, help="This worker's shard id in [0, num_shards).")
    ap.add_argument("--new_skill_out", default="", help="JSONL path for newly harvested skill candidates from this shard.")
    ap.add_argument("--audit_out", default="", help="JSONL path for per-use audit records from this shard.")
    ap.add_argument("--seed", type=int, default=1234, help="Seed for Python/NumPy/PyTorch before model initialization.")
    ap.add_argument("--deterministic_torch", action="store_true", help="Request deterministic PyTorch algorithms where available; may reduce speed.")

    args = ap.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    set_reproducible_seed(args.seed + args.shard_id, deterministic_torch=args.deterministic_torch)

    input_path = Path(args.input)
    detail_out = Path(args.detail_out)
    report_out = Path(args.report_out)
    skill_bank_dir = Path(args.skill_bank_dir)
    skill_bank_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(input_path))
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} test rows from {input_path}")
    print(f"[Parallel] shard_id={args.shard_id}, num_shards={args.num_shards}, frozen={args.freeze_skill_bank_during_round}")

    if args.mode == "round0_no_skill":
        skill_bank = SkillBank()
        retrieval_skill_bank = SkillBank() if args.freeze_skill_bank_during_round else skill_bank
        print("[Mode] round0_no_skill: start from empty skill bank")
    else:
        if args.skill_bank_in:
            bank_path = Path(args.skill_bank_in)
        else:
            bank_path = skill_bank_dir / "latest.jsonl"
        skill_bank = SkillBank.load(bank_path)
        retrieval_skill_bank = SkillBank.load(bank_path) if args.freeze_skill_bank_during_round else skill_bank
        print(f"[Mode] roundN_with_skill: loaded {len(skill_bank)} skills from {bank_path}")

    vlm = VLMRepromptModel(
        model_path=args.model,
        processor_path=args.processor_path or None,
        adapters=args.adapters,
        device=args.vlm_device,
        torch_dtype=args.torch_dtype,
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
    )
    evaluator = VoxTellEvaluator(args.voxtell_model_dir, device=args.seg_device)

    raw_dice_db = {}
    raw_dice_cache_path = Path(args.raw_dice_cache) if args.raw_dice_cache else None
    if raw_dice_cache_path and raw_dice_cache_path.exists():
        with raw_dice_cache_path.open("r", encoding="utf-8") as f:
            raw_dice_db = json.load(f)
            print(f"[Cache] Loaded {len(raw_dice_db)} raw dice results")

    case_metrics: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    added_skills = 0
    emitted_skill_candidates = 0
    emitted_audit_records = 0

    # Kept for shard-level debug report only. Global audit is replayed by merge script.
    skill_audit_stats = defaultdict(lambda: {"uses": 0, "pos_gain": 0.0, "neg_drop": 0.0})

    detail_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    new_skill_fw = None
    audit_fw = None
    if args.new_skill_out:
        new_skill_path = Path(args.new_skill_out)
        new_skill_path.parent.mkdir(parents=True, exist_ok=True)
        new_skill_fw = new_skill_path.open("w", encoding="utf-8")
    if args.audit_out:
        audit_path = Path(args.audit_out)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_fw = audit_path.open("w", encoding="utf-8")

    processed = 0
    try:
        with detail_out.open("w", encoding="utf-8") as fw:
            for source_idx, row in enumerate(rows):
                if (source_idx % args.num_shards) != args.shard_id:
                    continue
                processed += 1
                idx = source_idx + 1

                dataset = clean_text(row.get("dataset"))
                case_id = clean_text(row.get("case_id"))
                label_id = int(row["label_id"])
                label_name = clean_text(row.get("label_name"))
                img_path = clean_text(row.get("img_path"))
                seg_path = clean_text(row.get("seg_path"))
                raw_prompt = clean_text(row.get("raw_prompt"))
                case_dir = Path(img_path).parent
                pngs = collect_pngs(case_dir, args.png_subdir, args.num_images)

                if args.strict_num_images and len(pngs) != args.num_images:
                    continue
                if not pngs:
                    continue

                cache_key = f"{dataset}|{case_id}|{label_id}|{raw_prompt}"
                raw_cache_hit = cache_key in raw_dice_db
                raw_cache_legacy_format = False
                raw_dice: Optional[float] = None
                raw_seg_time_sec = float("nan")

                if raw_cache_hit:
                    raw_dice, raw_seg_time_sec, raw_cache_legacy_format = parse_raw_cache_entry(raw_dice_db[cache_key])

                raw_time_missing = not np.isfinite(raw_seg_time_sec)
                if raw_dice is None or (raw_time_missing and (not raw_cache_hit or args.recompute_missing_raw_time)):
                    raw_dice, raw_seg_time_sec = evaluator.evaluate_dice_timed(img_path, seg_path, label_id, raw_prompt)
                    raw_dice_db[cache_key] = make_raw_cache_entry(raw_dice, raw_seg_time_sec)
                    raw_cache_hit = False
                    raw_cache_legacy_format = False

                if raw_dice is None:
                    raise ValueError(f"Missing raw_dice for cache_key={cache_key}")

                rewrite_stage_start = time.perf_counter()

                retrieved_skills: List[Tuple[SkillArtifact, float]] = []
                candidate_skills: List[SkillArtifact] = []
                candidate_skill_scores: List[float] = []
                skill_bank_text = ""

                if args.mode == "roundN_with_skill" and len(retrieval_skill_bank) > 0:
                    retrieved_skills = retrieval_skill_bank.retrieve_topk_scored(row, args.skill_bank_topk)
                    filtered = [(s, sc) for s, sc in retrieved_skills if sc >= args.skill_use_min_score]
                    candidate_skills = [s for s, _ in filtered]
                    candidate_skill_scores = [float(sc) for _, sc in filtered]
                    if candidate_skills:
                        skill_bank_text = "\n\n".join(s.render_block(i + 1) for i, s in enumerate(candidate_skills))
                        user_prompt = args.user_prompt_with_skill
                    else:
                        user_prompt = args.user_prompt_no_skill
                else:
                    user_prompt = args.user_prompt_no_skill

                sync_cuda_if_available()
                vlm_t0 = time.perf_counter()
                full_response = vlm.generate_full_response(
                    image_paths=pngs,
                    raw_prompt=raw_prompt,
                    system_prompt=args.system_prompt,
                    user_prompt=user_prompt,
                    skill_bank_text=skill_bank_text,
                )
                sync_cuda_if_available()
                vlm_generate_time_sec = time.perf_counter() - vlm_t0

                ft_reprompt = parse_answer_text(full_response)
                if not ft_reprompt:
                    ft_reprompt = raw_prompt
                rewrite_stage_time_sec = time.perf_counter() - rewrite_stage_start

                ft_dice, ft_seg_time_sec = evaluator.evaluate_dice_timed(img_path, seg_path, label_id, ft_reprompt)
                ft_total_time_sec = float(rewrite_stage_time_sec + ft_seg_time_sec)
                latency_ratio_vs_raw_pct = (ft_total_time_sec / raw_seg_time_sec * 100.0) if raw_seg_time_sec > 0 else float("nan")
                latency_overhead_vs_raw_pct = latency_ratio_vs_raw_pct - 100.0 if np.isfinite(latency_ratio_vs_raw_pct) else float("nan")

                format_ok = has_valid_format(full_response)
                skill_gain = float(ft_dice - raw_dice)

                active_skills: List[SkillArtifact] = []
                if candidate_skills:
                    rat_m = RATIONALE_RE.search(full_response)
                    rat_text = rat_m.group(1).lower() if rat_m else ""

                    for i, s in enumerate(candidate_skills, 1):
                        if s.tag.lower() in rat_text or f"[{i}]" in rat_text or s.skill_id.lower() in rat_text:
                            active_skills.append(s)

                    if not active_skills:
                        active_skills = [candidate_skills[0]]

                    for s in active_skills:
                        skill_audit_stats[s.skill_id]["uses"] += 1
                        pos_gain = float(skill_gain) if skill_gain > 0 else 0.0
                        neg_drop = abs(float(skill_gain)) if skill_gain < 0 else 0.0
                        if skill_gain > 0:
                            skill_audit_stats[s.skill_id]["pos_gain"] += skill_gain
                        elif skill_gain < 0:
                            skill_audit_stats[s.skill_id]["neg_drop"] += abs(skill_gain)
                        if audit_fw is not None:
                            audit_fw.write(json.dumps({
                                "row_index": source_idx,
                                "skill_id": s.skill_id,
                                "uses": 1,
                                "pos_gain": pos_gain,
                                "neg_drop": neg_drop,
                            }, ensure_ascii=False) + "\n")
                            emitted_audit_records += 1

                skill_added = False
                if ft_dice >= args.skill_min_ft_dice and skill_gain >= args.skill_min_gain:
                    if (not args.skill_require_format) or format_ok:
                        new_skill = build_skill_from_pair(row, full_response, ft_reprompt, ft_dice, skill_gain)
                        skill_added = skill_bank.add(new_skill)
                        if skill_added:
                            added_skills += 1
                            if new_skill_fw is not None:
                                new_skill_fw.write(json.dumps({
                                    "row_index": source_idx,
                                    "skill": skill_to_row(new_skill),
                                }, ensure_ascii=False) + "\n")
                                emitted_skill_candidates += 1

                detail = {
                    "__row_index": source_idx,
                    "__shard_id": args.shard_id,
                    "__processed_index": processed,
                    "mode": args.mode,
                    "round_name": args.round_name,
                    "dataset": dataset,
                    "case_id": case_id,
                    "label_id": label_id,
                    "label_name": label_name,
                    "img_path": img_path,
                    "seg_path": seg_path,
                    "raw_prompt": raw_prompt,
                    "raw_dice": float(raw_dice),
                    "raw_cache_key": cache_key,
                    "raw_cache_hit": bool(raw_cache_hit),
                    "raw_cache_legacy_format": bool(raw_cache_legacy_format),
                    "raw_seg_time_sec": float(raw_seg_time_sec),
                    "rewrite_stage_time_sec": float(rewrite_stage_time_sec),
                    "vlm_generate_time_sec": float(vlm_generate_time_sec),
                    "ft_seg_time_sec": float(ft_seg_time_sec),
                    "ft_total_time_sec": float(ft_total_time_sec),
                    "latency_ratio_vs_raw_pct": float(latency_ratio_vs_raw_pct),
                    "latency_overhead_vs_raw_pct": float(latency_overhead_vs_raw_pct),
                    "retrieved_skill_count": len(retrieved_skills),
                    "used_skill_count": len(candidate_skills),
                    "skill_use_min_score": float(args.skill_use_min_score),
                    "retrieved_skills": [
                        {"skill_id": s.skill_id, "tag": s.tag, "content": s.content, "score": float(sc)}
                        for s, sc in retrieved_skills
                    ],
                    "used_skills": [
                        {"skill_id": s.skill_id, "tag": s.tag, "content": s.content, "score": float(sc)}
                        for s, sc in zip(candidate_skills, candidate_skill_scores)
                    ],
                    "full_response": full_response,
                    "format_ok": bool(format_ok),
                    "ft_reprompt": ft_reprompt,
                    "ft_dice": float(ft_dice),
                    "ft_gain": float(skill_gain),
                    "skill_added": bool(skill_added),
                    "num_pngs": len(pngs),
                }
                fw.write(json.dumps(detail, ensure_ascii=False) + "\n")
                if processed % 10 == 0:
                    fw.flush()
                    if new_skill_fw is not None:
                        new_skill_fw.flush()
                    if audit_fw is not None:
                        audit_fw.flush()

                case_key = (dataset, case_id, label_id, label_name)
                if case_key not in case_metrics:
                    case_metrics[case_key] = {
                        "dataset": dataset, "case_id": case_id, "label_id": label_id, "label_name": label_name,
                        "raw_dices": [], "ft_dices": [],
                    }
                case_metrics[case_key]["raw_dices"].append(float(raw_dice))
                case_metrics[case_key]["ft_dices"].append(float(ft_dice))

                print(
                    f"[{idx}/{len(rows)}|shard {args.shard_id}/{args.num_shards}|local {processed}] {args.mode} {dataset}/{case_id}/L{label_id} | "
                    f"raw={raw_dice:.4f} ft={ft_dice:.4f} gain={skill_gain:.4f} | "
                    f"lat={latency_ratio_vs_raw_pct:.2f}% raw_t={raw_seg_time_sec:.3f}s ft_total_t={ft_total_time_sec:.3f}s | "
                    f"reprompt={ft_reprompt}"
                )
    finally:
        if new_skill_fw is not None:
            new_skill_fw.close()
        if audit_fw is not None:
            audit_fw.close()

    if raw_dice_cache_path:
        raw_dice_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_dice_cache_path.open("w", encoding="utf-8") as f:
            json.dump(raw_dice_db, f, ensure_ascii=False, indent=2)

    # Shard-level debug report only. Final merged report is produced by merge_v5_frozen_parallel_repro.py.
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
    report_rows: List[Dict[str, Any]] = [group_report_rows(cr) for _, cr in sorted(grouped_case_rows.items())]
    write_csv(report_out, report_rows, ["dataset", "label_id", "label_name", "num_cases", "raw_mean", "ft_mean", "raw_std", "ft_std", "raw_worst", "ft_worst"])

    print(f"\n[Shard Summary] processed={processed}, local_added={added_skills}, emitted_skill_candidates={emitted_skill_candidates}, emitted_audit_records={emitted_audit_records}")
    print("[Shard Summary] No audit/fuse/cull/latest save is performed in worker; merge script finalizes the round.")


if __name__ == "__main__":
    main()
