#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib


def check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        print(f"[FAIL] {module_name}: {type(exc).__name__}: {exc}")
        return False
    print(f"[ OK ] {module_name}")
    return True


def main() -> None:
    modules = [
        "nibabel",
        "numpy",
        "torch",
        "transformers",
        "peft",
        "nnunetv2.imageio.nibabel_reader_writer",
        "voxtell.inference.predictor",
    ]
    ok = all(check_import(name) for name in modules)

    try:
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device count: {torch.cuda.device_count()}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] torch runtime check: {type(exc).__name__}: {exc}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
