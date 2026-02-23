#!/usr/bin/env python3
"""
Layer 2 – Probabilistic analysis of trained student checkpoints.

Evaluates one or more student checkpoints against the teacher on GSM8K
and saves scalar metrics + per-position curves to a JSON file, plus
comparison plots.

Usage
-----
Single checkpoint:
    python scripts/run_analysis.py \
        --config configs/kd_qwen_gsm8k.yaml \
        --checkpoint checkpoints/A_ce_only/final \
        --label CE

Multiple checkpoints (full comparison):
    python scripts/run_analysis.py \
        --config configs/kd_qwen_gsm8k.yaml \
        --checkpoints \
            student_base=Qwen/Qwen2.5-1.5B \
            CE=checkpoints/A_ce_only/final \
            KD=checkpoints/B_kd_only/final \
            KD+CE=checkpoints/C_kd_ce/final \
        --output-dir analysis_output

The config provides teacher name, teacher_load_mode, max_length, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_gsm8k import build_dataloader
from src.evaluate_probabilistic import evaluate_model, verify_same_tokenizer
from src.plotting_utils import plot_all
from src.train_manual import load_teacher, load_student


DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "kd_qwen_gsm8k.yaml")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[config] loaded from {path}")
    return cfg


def load_student_from_checkpoint(
    checkpoint_path: str,
    dtype_str: str = "bf16",
) -> AutoModelForCausalLM:
    """Load a student model from a local checkpoint or HF hub name.

    The model is set to **eval mode** with all parameters frozen.
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_single(
    label: str,
    checkpoint: str,
    teacher: AutoModelForCausalLM | None,
    dataloader: torch.utils.data.DataLoader,
    cfg: dict,
) -> dict:
    """Evaluate a single checkpoint and return the result dict."""
    print(f"\n{'─'*55}")
    print(f"  Evaluating: {label}  ({checkpoint})")
    print(f"{'─'*55}")

    student = load_student_from_checkpoint(
        checkpoint, cfg.get("student_dtype", "bf16"),
    )
    result = evaluate_model(student, teacher, dataloader, cfg)
    result["label"] = label
    result["checkpoint"] = checkpoint

    # Free student GPU memory
    del student
    torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 2 – Probabilistic analysis of student checkpoints",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to YAML config (provides teacher info, max_length, etc.)",
    )

    # ---- Single-model mode ----
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a single student checkpoint (or HF hub name).",
    )
    parser.add_argument(
        "--label", type=str, default="model",
        help="Label for the single checkpoint (e.g. 'CE', 'KD').",
    )

    # ---- Multi-model mode ----
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="label=path pairs, e.g.:  CE=checkpoints/A/final  KD=checkpoints/B/final",
    )

    # ---- Output ----
    parser.add_argument(
        "--output-dir", type=str, default="analysis_output",
        help="Directory for JSON results and plots.",
    )
    parser.add_argument(
        "--no-teacher", action="store_true",
        help="Skip teacher loading (no KL metrics).",
    )
    parser.add_argument(
        "--eval-split", type=str, default="test",
        help="GSM8K split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit evaluation to N batches (for quick debugging).",
    )
    parser.add_argument(
        "--smooth-window", type=int, default=5,
        help="Moving-average window for per-position plots (1=no smoothing).",
    )

    args = parser.parse_args()

    # ---- Load config ----
    cfg = load_config(args.config)

    # ---- Build checkpoint map ----
    ckpt_map: dict[str, str] = {}
    if args.checkpoints:
        for item in args.checkpoints:
            if "=" not in item:
                parser.error(
                    f"--checkpoints entries must be label=path, got: {item!r}"
                )
            lbl, path = item.split("=", 1)
            ckpt_map[lbl] = path
    elif args.checkpoint:
        ckpt_map[args.label] = args.checkpoint
    else:
        parser.error("Provide --checkpoint or --checkpoints.")

    # ---- Tokenizer ----
    tokenizer_name = cfg.get("student_name", list(ckpt_map.values())[0])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- DataLoader ----
    print(f"\n[data] Building eval dataloader (split={args.eval_split}) …")
    micro_n = cfg.get("micro_overfit_n", None)
    dataloader = build_dataloader(
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        batch_size=cfg.get("batch_size", 1),
        split=args.eval_split,
        micro_overfit_n=micro_n,
        shuffle=False,
    )

    # Optionally limit batches
    if args.max_batches is not None and args.max_batches > 0:
        print(f"[data] Limiting to {args.max_batches} batches.")
        from itertools import islice
        batches = list(islice(dataloader, args.max_batches))

        class _ListLoader:
            """Thin wrapper so we can iterate multiple times."""
            def __init__(self, items):
                self._items = items
            def __iter__(self):
                return iter(self._items)
            def __len__(self):
                return len(self._items)

        dataloader = _ListLoader(batches)

    # ---- Tokenizer consistency check ----
    if not args.no_teacher:
        print("\n[tokenizer] Verifying teacher/student tokenizer consistency …")
        verify_same_tokenizer(
            cfg["teacher_name"],
            cfg.get("student_name", tokenizer_name),
        )

    # ---- Teacher ----
    teacher = None
    if not args.no_teacher:
        print(f"\n[teacher] Loading {cfg['teacher_name']} "
              f"(mode={cfg['teacher_load_mode']}) …")
        teacher = load_teacher(cfg["teacher_name"], cfg["teacher_load_mode"])

    # ---- Evaluate each checkpoint ----
    all_results: dict[str, dict] = {}
    for label, ckpt_path in ckpt_map.items():
        result = run_single(label, ckpt_path, teacher, dataloader, cfg)
        all_results[label] = result

    # ---- Save JSON results ----
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[results] saved → {json_path}")

    # ---- Generate plots ----
    if len(all_results) >= 1:
        print("\n[plots] Generating comparison plots …")
        plot_all(all_results, args.output_dir, smooth_window=args.smooth_window)

    # ---- Summary table (global) ----
    print(f"\n{'='*85}")
    print("  GLOBAL SUMMARY")
    print(f"{'='*85}")
    header = (
        f"{'Model':<15} {'Entropy':>8} {'MaxProb':>8} "
        f"{'NLL':>8} {'PPL':>8} {'KL(T||S)':>9} {'ECE':>8} "
        f"{'Tokens':>8} {'AvgLen':>7}"
    )
    print(header)
    print("-" * len(header))
    for lbl, res in all_results.items():
        kl_str = f"{res['mean_kl']:.4f}" if res.get("mean_kl") is not None else "  n/a"
        print(
            f"{lbl:<15} {res['mean_entropy']:>8.4f} {res['mean_maxprob']:>8.4f} "
            f"{res['mean_nll']:>8.4f} {res['ppl']:>8.2f} "
            f"{kl_str:>9} {res['ece']:>8.4f} "
            f"{res['n_tokens_eval']:>8} {res['avg_seq_length']:>7.1f}"
        )
    print(f"{'='*85}")

    # ---- Summary table (per region) ----
    region_keys = [("Prompt", "region_prompt"),
                   ("Reasoning", "region_reasoning"),
                   ("Answer", "region_answer")]
    for rname, rkey in region_keys:
        # Check if any result has data for this region
        if not any(r.get(rkey, {}).get("n_tokens", 0) > 0 for r in all_results.values()):
            continue
        print(f"\n  {rname.upper()} REGION")
        rh = (
            f"  {'Model':<15} {'Entropy':>8} {'MaxProb':>8} "
            f"{'NLL':>8} {'PPL':>8} {'KL':>8} {'ECE':>8} {'Tokens':>8}"
        )
        print(rh)
        print("  " + "-" * (len(rh) - 2))
        for lbl, res in all_results.items():
            rd = res.get(rkey, {})
            rn = rd.get("n_tokens", 0)
            if rn == 0:
                print(f"  {lbl:<15} {'(no tokens)':>50}")
                continue
            rkl = f"{rd['kl']:.4f}" if rd.get("kl") is not None else "  n/a"
            print(
                f"  {lbl:<15} {rd['entropy']:>8.4f} {rd['maxprob']:>8.4f} "
                f"{rd['nll']:>8.4f} {rd['ppl']:>8.2f} "
                f"{rkl:>8} {rd['ece']:>8.4f} {rn:>8}"
            )

    print(f"\n{'='*85}\n")
    print("Done. ✓")


if __name__ == "__main__":
    main()
