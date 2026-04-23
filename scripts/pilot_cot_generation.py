#!/usr/bin/env python3
"""Pilot CoT generation for Article II (Phase 1.0).

Greedy generation of Chain-of-Thought (CoT) by the teacher model
(Qwen/Qwen2.5-7B-Instruct, bf16) on a sample of GSM8K ``train``
examples, to validate format adherence, length budget, and teacher
accuracy BEFORE the full generation run (Phase 1.1, 8 792 examples).

Gate criteria (Planejamento v2.2, Parte V, Fase 1.0):
    - separator ``####`` adherence   >= 95 %
    - mean(prompt + CoT + answer)    <= 750 tokens
    - percentile-95 total length     <= 768 tokens
    - teacher accuracy vs gold        >= 80 %

References
----------
Cobbe et al., 2021 — "Training Verifiers to Solve Math Word Problems"
    (GSM8K dataset; native ``\\n#### <number>`` separator).
Hsieh et al., 2023 — "Distilling Step-by-Step: Outperforming Larger
    Language Models with Less Training Data and Smaller Model Sizes"
    (few-shot CoT prompting of the teacher to obtain rationales).
Gu et al., 2024 — "MiniLLM: Knowledge Distillation of Large Language
    Models" (off-policy KD consumes fixed teacher outputs).

Usage
-----
    # Dry run with 5 samples (syntactic/functional check):
    python scripts/pilot_cot_generation.py --n-samples 5

    # Full pilot with 100 samples:
    python scripts/pilot_cot_generation.py --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

# Ensure project root is on sys.path (script can be run directly)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import datasets  # noqa: E402

from src.train_manual import load_teacher  # noqa: E402


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant that solves math word problems."

# Few-shot exemplars: first 4 indices of GSM8K train (deterministic,
# reproducible). The full native answer (including "#### N") is used as
# the assistant content — this enforces the target output format.
FEW_SHOT_INDICES: Tuple[int, ...] = (0, 1, 2, 3)

# ``####  <signed number, optional commas, optional decimal part>``
SEPARATOR_REGEX = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


# ------------------------------------------------------------------
# Answer extraction / comparison
# ------------------------------------------------------------------

def extract_gold_answer(gsm8k_answer: str) -> Optional[str]:
    """Extract the canonical gold number from a GSM8K ``answer`` string.

    The native GSM8K format ends with ``\\n#### <number>``.
    Returns the number as a normalised string (commas removed), or
    ``None`` if no ``####`` separator is found.
    """
    m = SEPARATOR_REGEX.search(gsm8k_answer)
    if m is None:
        return None
    return _normalise_number(m.group(1))


def extract_teacher_answer(generated_text: str) -> Tuple[bool, Optional[str]]:
    """Extract ``####``-delimited answer from teacher output.

    Returns ``(separator_found, extracted_answer_or_None)``.
    The first ``####`` occurrence is used; if no separator is present,
    ``(False, None)`` is returned.
    """
    m = SEPARATOR_REGEX.search(generated_text)
    if m is None:
        return False, None
    return True, _normalise_number(m.group(1))


def _normalise_number(s: str) -> str:
    """Remove thousands separators and strip whitespace."""
    return s.replace(",", "").strip()


def is_answer_correct(gold: Optional[str], pred: Optional[str]) -> bool:
    """Exact numeric match after normalisation.

    Attempts float comparison first (tolerates ``42`` vs ``42.0``);
    falls back to string equality.
    """
    if gold is None or pred is None:
        return False
    try:
        return float(gold) == float(pred)
    except (ValueError, TypeError):
        return gold == pred


# ------------------------------------------------------------------
# Few-shot prompt construction
# ------------------------------------------------------------------

def build_messages(
    train_ds: datasets.Dataset,
    question: str,
    few_shot_indices: Tuple[int, ...] = FEW_SHOT_INDICES,
) -> List[Dict[str, str]]:
    """Build the chat-template messages for the teacher.

    Structure:
        [system]
        [user: Q_0 ]  [assistant: A_0 (with ####) ]
        [user: Q_1 ]  [assistant: A_1            ]
        ...
        [user: Q_k ]  [assistant: A_k            ]
        [user: <question under test> ]            ← generation target

    The assistant turns in the few-shot block use the **native GSM8K
    answer** (full rationale + ``#### N``), which enforces the desired
    output format for the teacher. This is the "Distilling Step-by-Step"
    prompting style adapted to GSM8K's native separator.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for i in few_shot_indices:
        ex = train_ds[int(i)]
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


# ------------------------------------------------------------------
# Core pilot routine
# ------------------------------------------------------------------

def run_pilot(
    teacher_name: str = "Qwen/Qwen2.5-7B-Instruct",
    teacher_load_mode: str = "bf16",
    n_samples: int = 100,
    seed: int = 42,
    max_new_tokens: int = 512,
    output_dir: str = "outputs/pilot",
    few_shot_indices: Tuple[int, ...] = FEW_SHOT_INDICES,
) -> Dict[str, Any]:
    """Run the CoT-generation pilot and save ``pilot_results.json``.

    Returns the full results dict (also persisted to disk).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pilot_results.json")

    # ---- Reproducibility ----
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Load dataset ----
    print(f"[data] Loading GSM8K train split …")
    train_ds = datasets.load_dataset("gsm8k", "main", split="train")
    n_total = len(train_ds)
    print(f"[data] GSM8K train: {n_total} examples")

    # ---- Sample indices (excluding few-shot exemplars) ----
    rng = np.random.RandomState(seed)
    pool = np.array(
        [i for i in range(n_total) if i not in set(few_shot_indices)],
        dtype=np.int64,
    )
    if n_samples > len(pool):
        raise ValueError(
            f"n_samples={n_samples} exceeds pool size {len(pool)} "
            f"(GSM8K train minus few-shot indices)."
        )
    sample_indices = rng.choice(pool, size=n_samples, replace=False).tolist()
    sample_indices.sort()  # deterministic iteration order
    print(f"[sample] {n_samples} indices drawn (seed={seed}, "
          f"first 5: {sample_indices[:5]}) …")

    # ---- Tokenizer ----
    # The teacher tokenizer is what determines the prompt/generation
    # length budget during generation. We use it here consistently.
    print(f"[tokenizer] Loading tokenizer for {teacher_name} …")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Log how '####' is tokenised (Sanity A14 precursor, v2.2 check 14).
    sep_ids = tokenizer("####", add_special_tokens=False).input_ids
    print(f"[tokenizer] '####' → {len(sep_ids)} token(s): ids={sep_ids}")

    # ---- Load teacher ----
    print(f"[teacher] Loading {teacher_name} (mode={teacher_load_mode}) …")
    t0 = time.time()
    teacher = load_teacher(teacher_name, teacher_load_mode)
    print(f"[teacher] loaded in {time.time() - t0:.1f}s")
    device = next(teacher.parameters()).device

    # ---- Generation loop ----
    samples: List[Dict[str, Any]] = []
    t_gen_start = time.time()

    for rank, idx in enumerate(sample_indices):
        ex = train_ds[int(idx)]
        question: str = ex["question"]
        gold_answer_text: str = ex["answer"]
        gold = extract_gold_answer(gold_answer_text)

        # Build prompt with chat template
        messages = build_messages(train_ds, question, few_shot_indices)
        prompt_text: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenise prompt
        prompt_encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = prompt_encoded.input_ids.to(device)
        attention_mask = prompt_encoded.attention_mask.to(device)
        prompt_len = int(input_ids.shape[1])

        # Greedy generation
        with torch.inference_mode():
            gen_out = teacher.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_len = int(gen_out.shape[1])
        generation_len = full_len - prompt_len
        truncated = generation_len >= max_new_tokens

        # Decode generated portion only
        generated_ids = gen_out[0, prompt_len:]
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Separator + answer extraction
        sep_found, pred = extract_teacher_answer(generated_text)
        correct = is_answer_correct(gold, pred)

        sample_record = {
            "idx": int(idx),
            "question": question,
            "answer_gold": gold if gold is not None else "",
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "separator_found": bool(sep_found),
            "extracted_answer": pred,
            "is_correct": bool(correct),
            "total_len_tokens": full_len,
            "prompt_len_tokens": prompt_len,
            "generation_len_tokens": generation_len,
            "truncated_by_max_new_tokens": bool(truncated),
        }
        samples.append(sample_record)

        # Progress log
        if (rank + 1) % max(1, n_samples // 20) == 0 or (rank + 1) == n_samples:
            elapsed = time.time() - t_gen_start
            rate = (rank + 1) / max(elapsed, 1e-6)
            eta = (n_samples - (rank + 1)) / max(rate, 1e-6)
            print(
                f"[gen] {rank + 1:>4d}/{n_samples}  "
                f"idx={idx:>5d}  sep={sep_found}  correct={correct}  "
                f"gen_len={generation_len:>3d}  total={full_len:>3d}  "
                f"({rate:.2f} ex/s, ETA {eta / 60:.1f} min)"
            )

    # ---- Aggregate stats ----
    n = len(samples)
    sep_rate = sum(s["separator_found"] for s in samples) / max(n, 1)
    acc = sum(s["is_correct"] for s in samples) / max(n, 1)
    total_lens = np.array([s["total_len_tokens"] for s in samples], dtype=np.int64)
    gen_lens = np.array([s["generation_len_tokens"] for s in samples], dtype=np.int64)
    n_truncated = int((gen_lens >= max_new_tokens).sum())

    stats = {
        "n_samples": n,
        "separator_rate": float(sep_rate),
        "teacher_accuracy_vs_gold": float(acc),
        "mean_total_len_tokens": float(total_lens.mean()),
        "median_total_len_tokens": float(np.median(total_lens)),
        "p95_total_len_tokens": float(np.percentile(total_lens, 95)),
        "max_total_len_tokens": int(total_lens.max()),
        "mean_generation_len_tokens": float(gen_lens.mean()),
        "p95_generation_len_tokens": float(np.percentile(gen_lens, 95)),
        "max_generation_len_tokens": int(gen_lens.max()),
        "n_truncated_by_max_new_tokens": n_truncated,
    }

    # ---- Gate criteria (Fase 1.0) ----
    gates = {
        "separator_rate_>=_0.95": sep_rate >= 0.95,
        "mean_total_len_<=_750": float(total_lens.mean()) <= 750.0,
        "p95_total_len_<=_768": float(np.percentile(total_lens, 95)) <= 768.0,
        "teacher_accuracy_>=_0.80": acc >= 0.80,
    }
    all_passed = all(gates.values())

    # ---- Config echo ----
    config_echo = {
        "teacher_name": teacher_name,
        "teacher_load_mode": teacher_load_mode,
        "n_samples": n_samples,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "few_shot_indices": list(few_shot_indices),
        "system_prompt": SYSTEM_PROMPT,
        "separator_regex": SEPARATOR_REGEX.pattern,
        "sep_token_count_qwen25": len(sep_ids),
    }

    results = {
        "config": config_echo,
        "stats": stats,
        "gates": gates,
        "go_decision": bool(all_passed),
        "samples": samples,
    }

    # ---- Save ----
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] results → {out_path}")

    # ---- Terminal summary ----
    print("\n" + "=" * 70)
    print("  PILOT SUMMARY")
    print("=" * 70)
    print(f"  n_samples                 : {n}")
    print(f"  separator_rate            : {sep_rate:.3f}  (>= 0.95 ?)")
    print(f"  teacher_accuracy_vs_gold  : {acc:.3f}  (>= 0.80 ?)")
    print(f"  mean total_len (tokens)   : {total_lens.mean():.1f}  (<= 750 ?)")
    print(f"  p95  total_len (tokens)   : {np.percentile(total_lens, 95):.1f}  (<= 768 ?)")
    print(f"  max  total_len (tokens)   : {int(total_lens.max())}")
    print(f"  n_truncated @ max_new     : {n_truncated}")
    print("  ----")
    for k, v in gates.items():
        print(f"  [{'✓' if v else '✗'}] {k}")
    print(f"\n  decision (all gates)      : {'GO' if all_passed else 'NO-GO'}")
    print("=" * 70 + "\n")

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pilot CoT generation for Article II (Phase 1.0).",
    )
    p.add_argument("--teacher-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--teacher-load-mode", default="bf16",
                   choices=["bf16", "4bit", "8bit"])
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--output-dir", default="outputs/pilot")
    p.add_argument(
        "--few-shot-indices",
        default="0,1,2,3",
        help="Comma-separated GSM8K train indices for few-shot exemplars.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    few_shot = tuple(int(x) for x in args.few_shot_indices.split(",") if x.strip())
    run_pilot(
        teacher_name=args.teacher_name,
        teacher_load_mode=args.teacher_load_mode,
        n_samples=args.n_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        few_shot_indices=few_shot,
    )


if __name__ == "__main__":
    main()
