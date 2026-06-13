#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    "In <answer>, provide the final normalized segmentation prompt."
)

DEFAULT_USER_PROMPT = (
    "Review the provided medical image slices and the raw segmentation request. "
    "Return ONLY three parts (objective evidence, a short rationale, and the final normalized reprompt) in this exact format:"
    "\n<evidence>...</evidence>\n<rationale>...</rationale>\n<answer>...</answer>"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<evidence>.*?</evidence>\s*<rationale>.*?</rationale>\s*<answer>.*?</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
NUM_RE = re.compile(r"(\d+)")


def natural_key(s: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in NUM_RE.split(s)]


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
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class VLMRepromptModel:
    def __init__(
        self,
        model_path: str,
        processor_path: str = "",
        adapters: str = "",
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        attn_impl: str = "flash_attention_2",
        max_new_tokens: int = 256,
    ):
        self.model_path = model_path
        self.processor_path = processor_path or model_path
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
    ) -> str:
        content: List[Dict[str, Any]] = []
        for p in image_paths:
            content.append({"type": "image", "image": str(p)})
        content.append({"type": "text", "text": f"{user_prompt}\n\n<raw_prompt>{raw_prompt}</raw_prompt>"})
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

    def evaluate_dice(self, img_path: str, seg_path: str, label_id: int, prompt: str) -> float:
        img, _props = self.load_image(img_path)
        gt = (self.load_seg(seg_path) == int(label_id)).astype(np.uint8)

        sync_cuda_if_available()
        pred = self.predictor.predict_single_image(img, [prompt]).astype(np.uint8)
        if pred.ndim == 4:
            pred = pred[0]
        sync_cuda_if_available()

        return dice_binary(pred, gt)


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
        "rewrite_mean": safe_float_mean([r["case_rewrite_mean"] for r in case_rows]),
        "raw_std": safe_float_mean([r["case_raw_std"] for r in case_rows]),
        "rewrite_std": safe_float_mean([r["case_rewrite_std"] for r in case_rows]),
        "raw_worst": safe_float_mean([r["case_raw_worst"] for r in case_rows]),
        "rewrite_worst": safe_float_mean([r["case_rewrite_worst"] for r in case_rows]),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_reproducible_seed(seed: int, deterministic_torch: bool = False) -> None:
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
    ap.add_argument("--input", required=True, help="JSONL with dataset/case_id/img_path/seg_path/label_id/label_name/raw_prompt fields.")
    ap.add_argument("--model", required=True, help="Qwen3-VL base model path.")
    ap.add_argument("--adapters", default="", help="Optional LoRA adapter directory.")
    ap.add_argument("--processor_path", default="", help="Optional processor path; default uses --model.")
    ap.add_argument("--voxtell_model_dir", default="/home/dhm_41310/hdd/trzhang/models/VoxTell/voxtell_v1.1")
    ap.add_argument("--vlm_device", default="cuda:0")
    ap.add_argument("--seg_device", default="cuda:0")
    ap.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    ap.add_argument("--attn_impl", default="flash_attention_2")
    ap.add_argument("--png_subdir", default="vlm_png_axial")
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--strict_num_images", action="store_true")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--user_prompt", default=DEFAULT_USER_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detail_out", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--deterministic_torch", action="store_true")
    args = ap.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    set_reproducible_seed(args.seed + args.shard_id, deterministic_torch=args.deterministic_torch)

    input_path = Path(args.input)
    detail_out = Path(args.detail_out)
    report_out = Path(args.report_out)
    rows = list(iter_jsonl(input_path))
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"[Parallel] shard_id={args.shard_id}, num_shards={args.num_shards}")

    vlm = VLMRepromptModel(
        model_path=args.model,
        processor_path=args.processor_path,
        adapters=args.adapters,
        device=args.vlm_device,
        torch_dtype=args.torch_dtype,
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
    )
    evaluator = VoxTellEvaluator(args.voxtell_model_dir, device=args.seg_device)

    detail_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    case_metrics: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    processed = 0
    skipped_no_png = 0

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
            pngs = collect_pngs(Path(img_path).parent, args.png_subdir, args.num_images)

            if args.strict_num_images and len(pngs) != args.num_images:
                skipped_no_png += 1
                continue
            if not pngs:
                skipped_no_png += 1
                continue

            raw_dice = evaluator.evaluate_dice(img_path, seg_path, label_id, raw_prompt)

            sync_cuda_if_available()
            full_response = vlm.generate_full_response(
                image_paths=pngs,
                raw_prompt=raw_prompt,
                system_prompt=args.system_prompt,
                user_prompt=args.user_prompt,
            )
            sync_cuda_if_available()

            rewrite_prompt = parse_answer_text(full_response) or raw_prompt
            rewrite_dice = evaluator.evaluate_dice(img_path, seg_path, label_id, rewrite_prompt)
            gain = float(rewrite_dice - raw_dice)

            detail = {
                "__row_index": source_idx,
                "__shard_id": args.shard_id,
                "__processed_index": processed,
                "dataset": dataset,
                "case_id": case_id,
                "label_id": label_id,
                "label_name": label_name,
                "img_path": img_path,
                "seg_path": seg_path,
                "raw_prompt": raw_prompt,
                "raw_dice": float(raw_dice),
                "full_response": full_response,
                "format_ok": bool(has_valid_format(full_response)),
                "rewrite_prompt": rewrite_prompt,
                "rewrite_dice": float(rewrite_dice),
                "rewrite_gain": float(gain),
                "num_pngs": len(pngs),
            }
            fw.write(json.dumps(detail, ensure_ascii=False) + "\n")
            fw.flush()

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
            case_metrics[case_key]["raw_dices"].append(float(raw_dice))
            case_metrics[case_key]["rewrite_dices"].append(float(rewrite_dice))

            print(
                f"[{idx}/{len(rows)}|shard {args.shard_id}/{args.num_shards}|local {processed}] "
                f"{dataset}/{case_id}/L{label_id} | raw={raw_dice:.4f} rewrite={rewrite_dice:.4f} "
                f"gain={gain:.4f} | "
                f"prompt={rewrite_prompt}"
            )

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
    report_rows = [group_report_rows(cr) for _, cr in sorted(grouped_case_rows.items())]
    write_csv(
        report_out,
        report_rows,
        ["dataset", "label_id", "label_name", "num_cases", "raw_mean", "rewrite_mean", "raw_std", "rewrite_std", "raw_worst", "rewrite_worst"],
    )

    print(f"\n[Shard Summary] processed={processed}, skipped_no_png={skipped_no_png}, written={sum(len(v['raw_dices']) for v in case_metrics.values())}")


if __name__ == "__main__":
    main()
