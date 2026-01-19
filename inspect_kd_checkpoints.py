from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset

from config import EvidenceBasedConfig, GenerationConfig, resolve_device, set_seed
from eval import _batched_generate_continuations, extract_gsm8k_answer
from evaluate_saved_models import _iter_model_dirs, _load_model_and_tokenizer


@dataclass
class GSM8KDebugStats:
    accuracy: float
    correct: int
    total: int
    empty_pred: int
    mean_gen_tokens: float
    has_hash_hash_hash_hash_rate: float
    has_final_answer_marker_rate: float


def _boolish(v: str) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_epochish(p: Path) -> int:
    # Tries to order dirs like epoch_01, step_0500, etc.
    name = p.name
    m = re.search(r"(?:epoch|ep)[^0-9]*(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:step|iter)[^0-9]*(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: any number in name.
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    return 10**9


def _gsm8k_prompts(questions: List[str], *, use_cot_prompt: bool) -> List[str]:
    if use_cot_prompt:
        return [f"Q: {q}\nA: Let's think step by step." for q in questions]
    return [f"Q: {q}\nA:" for q in questions]


def _debug_gsm8k(
    model,
    tokenizer,
    *,
    device: torch.device,
    seed: int,
    max_length: int,
    eval_limit: int,
    gen_cfg: GenerationConfig,
    use_cot_prompt: bool,
    batch_size: int,
    dump_samples: int,
) -> Tuple[GSM8KDebugStats, List[Dict[str, Any]]]:
    ds = load_dataset("gsm8k", "main", split="test")
    limit = min(int(eval_limit), len(ds))
    rng = np.random.RandomState(int(seed))
    indices = rng.choice(len(ds), limit, replace=False)

    questions: List[str] = []
    golds: List[str] = []
    for idx in indices:
        ex = ds[int(idx)]
        questions.append(str(ex.get("question", "")))
        # GSM8K gold uses "####" in the official answers.
        golds.append(str(ex.get("answer", "") or ex.get("output", "")))

    prompts = _gsm8k_prompts(questions, use_cot_prompt=bool(use_cot_prompt))

    conts, cont_lens = _batched_generate_continuations(
        model,
        tokenizer,
        prompts,
        device=device,
        max_length=int(max_length),
        gen_cfg=gen_cfg,
        max_new_tokens=int(gen_cfg.max_new_tokens),
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
        batch_size=int(batch_size),
    )

    correct = 0
    total = 0
    empty_pred = 0
    has_hash = 0
    has_final_marker = 0

    samples: List[Dict[str, Any]] = []

    for q, cont, gold, clen in zip(questions, conts, golds, cont_lens):
        pred = extract_gsm8k_answer(cont)
        gold_ex = extract_gsm8k_answer(gold)

        if not pred:
            empty_pred += 1

        if "####" in (cont or ""):
            has_hash += 1
        if re.search(r"###\s*FINAL_ANSWER", cont or "", flags=re.IGNORECASE):
            has_final_marker += 1

        if pred == gold_ex and gold_ex != "":
            correct += 1
        total += 1

        if dump_samples and len(samples) < int(dump_samples):
            samples.append(
                {
                    "question": q,
                    "gold_extracted": gold_ex,
                    "pred_extracted": pred,
                    "generated_tokens": int(clen),
                    "continuation": cont,
                }
            )

    mean_gen = float(np.mean([int(x) for x in cont_lens])) if cont_lens else 0.0

    stats = GSM8KDebugStats(
        accuracy=(correct / total) if total else 0.0,
        correct=int(correct),
        total=int(total),
        empty_pred=int(empty_pred),
        mean_gen_tokens=float(mean_gen),
        has_hash_hash_hash_hash_rate=(float(has_hash) / float(total)) if total else 0.0,
        has_final_answer_marker_rate=(float(has_final_marker) / float(total)) if total else 0.0,
    )

    return stats, samples


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Inspect KD checkpoints to find where GSM8K starts to drift. "
            "Loads each checkpoint dir (config.json or adapter_config.json), runs a small GSM8K eval, "
            "and reports parsing/format diagnostics."
        )
    )

    p.add_argument("--models_root", type=str, required=True, help="Root directory containing checkpoints/model dirs.")
    p.add_argument("--recursive", action="store_true", help="Search recursively under --models_root.")
    p.add_argument("--teacher_dir", type=str, default=None, help="Optional teacher checkpoint dir for side-by-side GSM8K debug.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu, cuda, cuda:0 (default: auto)")
    p.add_argument("--load_dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")

    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--eval_limit_gsm8k", type=int, default=50)
    p.add_argument("--use_cot_prompt", action="store_true")
    p.add_argument("--no_cot_prompt", action="store_true")

    p.add_argument("--eval_max_new_tokens", type=int, default=256)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--eval_do_sample", action="store_true")
    p.add_argument("--eval_repetition_penalty", type=float, default=1.1)

    p.add_argument("--batch_size", type=int, default=None, help="Eval micro-batch size (default: env SLM_EVAL_BATCH_SIZE or 4)")
    p.add_argument("--dump_samples", type=int, default=0, help="Dump N sample generations per checkpoint.")

    p.add_argument("--out_json", type=str, default=None, help="If set, write a JSON report here.")

    args = p.parse_args(argv)

    set_seed(int(args.seed))

    device = resolve_device(args.device)
    use_cot_prompt = True
    if args.use_cot_prompt:
        use_cot_prompt = True
    if args.no_cot_prompt:
        use_cot_prompt = False

    batch_size = int(args.batch_size or os.environ.get("SLM_EVAL_BATCH_SIZE", "4") or 4)

    gen_cfg = GenerationConfig(
        max_new_tokens=int(args.eval_max_new_tokens),
        temperature=float(args.eval_temperature),
        do_sample=bool(args.eval_do_sample),
        repetition_penalty=float(args.eval_repetition_penalty),
    )

    # Find checkpoint dirs.
    model_dirs = _iter_model_dirs([str(args.models_root)], recursive=bool(args.recursive))
    model_dirs = sorted(model_dirs, key=_extract_epochish)

    if not model_dirs:
        raise SystemExit(f"No checkpoint/model dirs found under: {args.models_root}")

    # Optional teacher baseline.
    teacher_stats: Optional[GSM8KDebugStats] = None
    teacher_samples: List[Dict[str, Any]] = []
    if args.teacher_dir:
        tdir = Path(args.teacher_dir)
        tmodel, ttok = _load_model_and_tokenizer(tdir, device, str(args.load_dtype))
        teacher_stats, teacher_samples = _debug_gsm8k(
            tmodel,
            ttok,
            device=device,
            seed=int(args.seed),
            max_length=int(args.max_length),
            eval_limit=int(args.eval_limit_gsm8k),
            gen_cfg=gen_cfg,
            use_cot_prompt=bool(use_cot_prompt),
            batch_size=int(batch_size),
            dump_samples=int(args.dump_samples),
        )
        print("\n[teacher]", str(tdir))
        print(
            f"  gsm8k_acc={teacher_stats.accuracy:.4f} empty_pred={teacher_stats.empty_pred}/{teacher_stats.total} "
            f"mean_gen_tok={teacher_stats.mean_gen_tokens:.1f} "
            f"has####={teacher_stats.has_hash_hash_hash_hash_rate:.2%} hasFINAL={teacher_stats.has_final_answer_marker_rate:.2%}"
        )

    rows: List[Dict[str, Any]] = []

    for d in model_dirs:
        print("\n[ckpt]", str(d))
        model, tok = _load_model_and_tokenizer(d, device, str(args.load_dtype))

        stats, samples = _debug_gsm8k(
            model,
            tok,
            device=device,
            seed=int(args.seed),
            max_length=int(args.max_length),
            eval_limit=int(args.eval_limit_gsm8k),
            gen_cfg=gen_cfg,
            use_cot_prompt=bool(use_cot_prompt),
            batch_size=int(batch_size),
            dump_samples=int(args.dump_samples),
        )

        print(
            f"  gsm8k_acc={stats.accuracy:.4f} empty_pred={stats.empty_pred}/{stats.total} "
            f"mean_gen_tok={stats.mean_gen_tokens:.1f} "
            f"has####={stats.has_hash_hash_hash_hash_rate:.2%} hasFINAL={stats.has_final_answer_marker_rate:.2%}"
        )

        if samples:
            print("  samples:")
            for s in samples:
                pred = s.get("pred_extracted")
                gold = s.get("gold_extracted")
                print(f"    - pred={pred!r} gold={gold!r} gen_tok={s.get('generated_tokens')}")

        rows.append(
            {
                "model_dir": str(d),
                "epochish": _extract_epochish(d),
                "gsm8k": {
                    "accuracy": stats.accuracy,
                    "correct": stats.correct,
                    "total": stats.total,
                    "empty_pred": stats.empty_pred,
                    "mean_generated_tokens": stats.mean_gen_tokens,
                    "has_hash_rate": stats.has_hash_hash_hash_hash_rate,
                    "has_final_answer_marker_rate": stats.has_final_answer_marker_rate,
                },
                "samples": samples,
            }
        )

        # Free between checkpoints (important on GPU).
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    out: Dict[str, Any] = {
        "seed": int(args.seed),
        "models_root": str(args.models_root),
        "recursive": bool(args.recursive),
        "device": str(device),
        "use_cot_prompt": bool(use_cot_prompt),
        "eval_limit_gsm8k": int(args.eval_limit_gsm8k),
        "generation": {
            "max_new_tokens": int(gen_cfg.max_new_tokens),
            "temperature": float(gen_cfg.temperature),
            "do_sample": bool(gen_cfg.do_sample),
            "repetition_penalty": float(gen_cfg.repetition_penalty or 1.0),
        },
        "teacher": (
            {
                "teacher_dir": str(args.teacher_dir),
                "gsm8k": teacher_stats.__dict__ if teacher_stats else None,
                "samples": teacher_samples,
            }
            if args.teacher_dir
            else None
        ),
        "checkpoints": rows,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
