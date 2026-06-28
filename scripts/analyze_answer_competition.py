#!/usr/bin/env python3
"""
Answer-region top-2 competition analysis (reviewer W2).

ECE in this paper treats the top-1 token as the prediction and its probability
as the confidence. Reviewer W2 asks: how often does a SECOND token also carry
non-trivial probability mass in the answer region? If almost never, the top-1
ECE adaptation is a tight approximation; where competition is high, answer-phase
ECE should be read as an upper bound.

This script runs a teacher-forced forward pass of one student checkpoint over the
GSM8K-CoT test set and reports, per region (reasoning / answer), the fraction of
valid tokens whose 2nd-highest probability exceeds {0.05, 0.10, 0.20}. It also
reports the mean 2nd-prob and mean (p1 - p2) margin.

No existing file is modified. Student-only (teacher not needed).

Usage
-----
    # Trained checkpoint
    python scripts/analyze_answer_competition.py \
        --config configs/gsm8k/gsm8k_cot_fkl_T4_seed42.yaml \
        --checkpoint checkpoints/gsm8k_cot_fkl_T4_seed42/final \
        --label FKL_T4

    # Base (untrained) student - same flag accepts an HF hub name
    python scripts/analyze_answer_competition.py \
        --config configs/gsm8k/gsm8k_cot_ce_only_T1_seed42.yaml \
        --checkpoint Qwen/Qwen2.5-1.5B-Instruct \
        --label BASE
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.data_gsm8k import (
    build_dataloader_cot,
    REGION_REASONING,
    REGION_ANSWER,
)
from src.losses.losses_kd import shift_for_causal_lm

THRESHOLDS = (0.05, 0.10, 0.20)
REGION_NAMES = {REGION_REASONING: "reasoning", REGION_ANSWER: "answer"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer-region top-2 competition (W2)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Local checkpoint path OR HF hub name (e.g. base student).")
    ap.add_argument("--label", default="model")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Cap batches for a quick check (default: full split).")
    ap.add_argument("--output", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg["student_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] student <- {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16,
    ).to(device).eval()

    loader = build_dataloader_cot(
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
        jsonl_path=cfg["cot_data_path"],
        split=args.split,
        filter_teacher_wrong=cfg.get("filter_teacher_wrong", False),
        shuffle=False,
        seed=cfg.get("seed", 42),
    )

    # Per-region accumulators
    agg = {
        rid: {
            "n": 0,
            "gt": {thr: 0 for thr in THRESHOLDS},
            "sum_p2": 0.0,
            "sum_margin": 0.0,
        }
        for rid in REGION_NAMES
    }

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            shift_logits, _, valid_mask = shift_for_causal_lm(
                logits, batch["labels"], batch["attention_mask"],
            )
            shift_region = batch["region_ids"][:, 1:].contiguous()

            probs = F.softmax(shift_logits.float(), dim=-1)         # [B, L-1, V]
            top2 = probs.topk(2, dim=-1).values                    # [B, L-1, 2]
            p1 = top2[..., 0]
            p2 = top2[..., 1]
            margin = p1 - p2

            for rid in REGION_NAMES:
                rmask = valid_mask & (shift_region == rid)
                n = int(rmask.sum().item())
                if n == 0:
                    continue
                p2_r = p2[rmask]
                agg[rid]["n"] += n
                agg[rid]["sum_p2"] += float(p2_r.sum().item())
                agg[rid]["sum_margin"] += float(margin[rmask].sum().item())
                for thr in THRESHOLDS:
                    agg[rid]["gt"][thr] += int((p2_r > thr).sum().item())

    # Report
    result = {"label": args.label, "checkpoint": args.checkpoint, "regions": {}}
    print(f"\n=== Top-2 competition: {args.label} ({args.split}) ===")
    for rid, name in REGION_NAMES.items():
        a = agg[rid]
        n = a["n"]
        if n == 0:
            continue
        row = {
            "n_tokens": n,
            "mean_p2": a["sum_p2"] / n,
            "mean_margin_p1_minus_p2": a["sum_margin"] / n,
            "frac_p2_gt": {str(thr): a["gt"][thr] / n for thr in THRESHOLDS},
        }
        result["regions"][name] = row
        print(f"\n[{name}]  n={n:,}")
        print(f"  mean 2nd-prob          = {row['mean_p2']:.4f}")
        print(f"  mean margin (p1 - p2)  = {row['mean_margin_p1_minus_p2']:.4f}")
        for thr in THRESHOLDS:
            print(f"  frac with p2 > {thr:.2f}    = {row['frac_p2_gt'][str(thr)]:.4%}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[save] -> {args.output}")

    print(
        "\nReviewer note (W2): report the answer-region 'frac with p2 > 0.10'. "
        "A small value bounds how much the top-1 ECE adaptation can overstate "
        "answer-phase miscalibration."
    )


if __name__ == "__main__":
    main()
