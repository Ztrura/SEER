#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor


DEFAULT_INPUT_DIR = "/home/dhm_41310/trzhang/SEER/OS/test_scripts/input/006_image.nii.gz"
DEFAULT_OUTPUT_DIR_RAW = "/home/dhm_41310/trzhang/SEER/OS/test_scripts/output/raw_out.nii.gz"
DEFAULT_OUTPUT_DIR_SEER = "/home/dhm_41310/trzhang/SEER/OS/test_scripts/output/seer_out.nii.gz"
DEFAULT_VOXTELL_MODEL_DIR = "/home/dhm_41310/hdd/trzhang/models/VoxTell/voxtell_v1.1"
DEFAULT_MODEL_DIR = "/home/dhm_41310/hdd/trzhang/models/Qwen3-VL-4B-Instruct"
DEFAULT_ADAPTER_DIR = "/home/dhm_41310/hdd/trzhang/models/SEER/GRPO_VLM/v9-20260418-135553/checkpoint-4099"
DEFAULT_PROMPT = "I need the right hipbone segmented, specifically the iliac part."

ROOT = Path(__file__).resolve().parents[1]
INFER_V1_1_DIR = ROOT / "infer" / "v1_1"
sys.path.insert(0, str(INFER_V1_1_DIR))


from infer_v1_1 import ( 
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_NO_SKILL,
    VLMRepromptModel,
    collect_pngs,
    parse_answer_text,
)


def load_image_for_voxtell(io: NibabelIOWithReorient, img_path: Path) -> Tuple[np.ndarray, Any]:
    img, props = io.read_images([str(img_path)])
    if isinstance(props, (list, tuple)) and len(props) == 1:
        props = props[0]
    return img, props


def save_mask_with_voxtell_io(
    io: NibabelIOWithReorient,
    mask: np.ndarray,
    output_path: Path,
    props: Any,
) -> None:
    out = np.asarray(mask).astype(np.uint8)
    if out.ndim == 4 and out.shape[0] == 1:
        out = out[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.write_seg(out, str(output_path), props)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM reprompt + VoxTell inference for one NIfTI case.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input image, e.g. <path>/input/FLARE22_Tr_0001_CT.nii.gz")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR_RAW, help="Output mask, e.g. <path>/output/01_out.nii.gz")
    parser.add_argument("--output_seer", default=DEFAULT_OUTPUT_DIR_SEER, help="Output mask, e.g. <path>/output/01_out.nii.gz")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Raw text prompt for the target structure, e.g. 'liver'.")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help="Qwen3-VL model/checkpoint path for reprompting.")
    parser.add_argument("--adapters", default=DEFAULT_ADAPTER_DIR, help="Optional LoRA adapter directory.")
    parser.add_argument("--processor_path", default="", help="Optional processor path; default uses --model.")
    parser.add_argument("--voxtell_model_dir", default=DEFAULT_VOXTELL_MODEL_DIR)
    parser.add_argument("--vlm_device", default="cuda:0")
    parser.add_argument("--seg_device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    parser.add_argument("--attn_impl", default="flash_attention_2")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--png_subdir", default="vlm_png_axial")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user_prompt", default=DEFAULT_USER_PROMPT_NO_SKILL)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path_seer = Path(args.output_seer)
    pngs = collect_pngs(input_path.parent, args.png_subdir, args.num_images)
    if not pngs:
        raise FileNotFoundError(f"No PNG slices found under {input_path.parent / args.png_subdir}")

    vlm = VLMRepromptModel(
        model_path=args.model,
        processor_path=args.processor_path or None,
        adapters=args.adapters,
        device=args.vlm_device,
        torch_dtype=args.torch_dtype,
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
    )
    full_response = vlm.generate_full_response(
        image_paths=pngs,
        raw_prompt=args.prompt,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        skill_bank_text="",
    )
    reprompt = parse_answer_text(full_response) or args.prompt

    io = NibabelIOWithReorient()
    img, props = load_image_for_voxtell(io, input_path)
    seg_device = torch.device(args.seg_device if torch.cuda.is_available() else "cpu")
    predictor = VoxTellPredictor(model_dir=args.voxtell_model_dir, device=seg_device)
    pred = predictor.predict_single_image(img, [args.prompt]).astype(np.uint8)
    pred_seer = predictor.predict_single_image(img, [reprompt]).astype(np.uint8)

    save_mask_with_voxtell_io(io, pred, output_path, props)
    save_mask_with_voxtell_io(io, pred_seer, output_path_seer, props)
    print(f"raw_prompt: {args.prompt}")
    print(f"reprompt: {reprompt}")
    print(f"saved: {output_path} & {output_path_seer}")


if __name__ == "__main__":
    main()
