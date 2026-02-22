#!/usr/bin/env python3
"""
Entry point: load config → (optional) sanity checks → train.

Usage
-----
    python scripts/run.py                              # uses default config
    python scripts/run.py --config configs/kd_qwen_gsm8k.yaml
"""

from __future__ import annotations

import argparse
import sys
import os

import yaml

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_manual import load_teacher, load_student, train
from src.sanity import run_all_sanity_checks
from transformers import AutoTokenizer


DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "kd_qwen_gsm8k.yaml")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded from {path}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="KD training runner")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Optional sanity checks ----
    if cfg.get("run_sanity", False):
        print("\n>>> Running sanity checks …\n")

        tokenizer = AutoTokenizer.from_pretrained(cfg["student_name"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        teacher = load_teacher(cfg["teacher_name"], cfg["teacher_load_mode"])
        student = load_student(cfg["student_name"], cfg["student_dtype"])

        run_all_sanity_checks(teacher, student, tokenizer, cfg)

        # Free sanity models to reclaim memory before training
        del teacher, student, tokenizer
        import torch
        torch.cuda.empty_cache()

        print("\n>>> Sanity checks done – proceeding to training\n")

    # ---- Train ----
    train(cfg)


if __name__ == "__main__":
    main()
