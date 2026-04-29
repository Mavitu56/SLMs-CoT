#!/usr/bin/env python3
"""Full CoT generation for Article II (Phase 1.1).

Greedy generation of teacher CoT for the full GSM8K dataset
(7 473 train + 1 319 test = 8 792 examples). Produces
``data/gsm8k_cot_qwen25_7b.jsonl`` — one JSON line per example —
consumed by ``src/data_gsm8k.py`` in Phase 2 via
``load_gsm8k_cot_jsonl`` + ``tokenise_example_cot``.

Design (v2.2 §V.1.1, session decisions 2026-04-24)
---------------------------------------------------
* **Same system prompt, few-shot, decoding as the validated pilot.**
  ``scripts/pilot_cot_generation.py`` is the canonical artefact
  (commits 9a1d21b, f119fd0). This script imports its helpers —
  ``SYSTEM_PROMPT``, ``FEW_SHOT_INDICES``, ``build_messages``,
  ``extract_teacher_answer``, ``extract_boxed_fallback``,
  ``extract_gold_answer``, ``is_answer_correct`` — verbatim. No
  logic is duplicated.
* **Few-shot coverage (decision 1a).** Indices
  ``{0, 1, 2, 3}`` are excluded from the generation pool on the
  ``train`` split. These examples serve as few-shot exemplars in
  every prompt; they are not distillation targets. Net train
  yield: 7 469 lines. Test split is always fully generated
  (1 319 lines). Total JSONL size: **8 788 lines**.
* **Resumable writes (decision 2a).** Output is opened in append
  mode. Before writing, the script reads any pre-existing JSONL
  and collects ``(split, idx)`` pairs already processed; these are
  skipped. A single run can therefore be restarted after a Colab
  disconnect without duplicating work or corrupting state.
* **Student tokenizer for length budget (decision 3a).** The JSONL
  is input to student training (``tokenise_example_cot`` with
  ``max_length=1024``). ``total_len_tokens`` is measured under the
  **student** tokenizer (Qwen2.5-1.5B-Instruct) — the budget the
  downstream loader will actually enforce. Teacher generation
  itself still uses the teacher tokenizer for correctness of
  ``model.generate``; the two vocabularies share identical token
  ids on the common prefix (verified by ``load_student`` checks in
  ``src/train_manual.py``), so this is a labelling choice, not a
  re-tokenisation.

Output line schema
------------------
Exactly the fields mandated by v2.2 §V.1.1::

    {
      "split": "train" | "test",
      "idx": int,
      "question": str,
      "answer_gold": str,
      "teacher_full_text": str,
      "extracted_answer": str | null,
      "is_teacher_correct": bool,
      "separator_found": bool,
      "total_len_tokens": int
    }

Also produces a sidecar ``data/cot_stats.json`` (Tarefa 1.1.3)
with aggregate statistics per split.

References
----------
Cobbe et al., 2021 — GSM8K (native ``\\n#### <number>`` separator).
Hsieh et al., 2023 — Distilling Step-by-Step (teacher-generated
    rationales as supervision for the student).
Gu et al., 2024 — MiniLLM (off-policy KD consumes fixed teacher
    outputs; consistency of ``P_teacher`` across the dataset is
    critical for the FKL vs RKL comparison in Article II).

Usage
-----
    # Full run (takes ~2-4 h on an A100 bf16):
    python scripts/generate_full_cot.py

    # Restart after a disconnect (resumes automatically):
    python scripts/generate_full_cot.py

    # Smoke test on a tiny prefix (no overwrite of the real output):
    python scripts/generate_full_cot.py \\
        --output-path data/gsm8k_cot_qwen25_7b.smoke.jsonl \\
        --max-per-split 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Set, Tuple

import torch
from transformers import AutoTokenizer

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import datasets  # noqa: E402

from src.training.train_manual import load_teacher  # noqa: E402
# Import validated helpers from the pilot — do NOT duplicate logic.
from scripts.pilot_cot_generation import (  # noqa: E402
    FEW_SHOT_INDICES,
    SYSTEM_PROMPT,
    build_messages,
    extract_boxed_fallback,
    extract_gold_answer,
    extract_teacher_answer,
    is_answer_correct,
)


DEFAULT_STUDENT_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TEACHER_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_PATH = "data/gsm8k_cot_qwen25_7b.jsonl"
DEFAULT_STATS_PATH = "data/cot_stats.json"


# ------------------------------------------------------------------
# Resume support
# ------------------------------------------------------------------

def load_done_keys(jsonl_path: str) -> Set[Tuple[str, int]]:
    """Return the set of ``(split, idx)`` already present in ``jsonl_path``.

    Ignores unparseable lines (e.g. a trailing half-written line from
    a previous crash) and logs their count. The caller decides what
    to do with a non-empty count — here we simply proceed, since the
    corrupt tail will be appended past on the next write.
    """
    if not os.path.exists(jsonl_path):
        return set()

    done: Set[Tuple[str, int]] = set()
    n_bad = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((str(rec["split"]), int(rec["idx"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                n_bad += 1
    if n_bad:
        print(f"[resume] skipped {n_bad} malformed line(s) in {jsonl_path}")
    print(f"[resume] {len(done)} example(s) already present in {jsonl_path}")
    return done


# ------------------------------------------------------------------
# Single-example generation
# ------------------------------------------------------------------

def generate_one(
    split: str,
    idx: int,
    example: Dict[str, Any],
    train_ds: datasets.Dataset,
    teacher,
    teacher_tokenizer: AutoTokenizer,
    student_tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Run the teacher on a single example and return the JSONL record.

    The few-shot context is always drawn from the ``train`` split at
    ``FEW_SHOT_INDICES`` (decision 1a / v2.2). ``total_len_tokens`` is
    measured with the **student** tokenizer (decision 3a), since the
    JSONL will be consumed by the student training loop.
    """
    question: str = example["question"]
    gold_answer_text: str = example["answer"]
    gold = extract_gold_answer(gold_answer_text)

    # Build prompt with the teacher tokenizer's chat template.
    messages = build_messages(train_ds, question, FEW_SHOT_INDICES)
    prompt_text: str = teacher_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_encoded = teacher_tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = prompt_encoded.input_ids.to(device)
    attention_mask = prompt_encoded.attention_mask.to(device)

    with torch.inference_mode():
        gen_out = teacher.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            pad_token_id=teacher_tokenizer.eos_token_id,
        )

    prompt_len_teacher = int(input_ids.shape[1])
    generated_ids = gen_out[0, prompt_len_teacher:]
    teacher_full_text = teacher_tokenizer.decode(
        generated_ids, skip_special_tokens=True,
    )

    # Format / correctness extraction (same helpers as the pilot).
    sep_found, pred = extract_teacher_answer(teacher_full_text)
    boxed_pred = extract_boxed_fallback(teacher_full_text)
    latent_pred = pred if pred is not None else boxed_pred
    latent_correct = is_answer_correct(gold, latent_pred)
    strict_correct = is_answer_correct(gold, pred)

    # v2.2 §V.1.1 specifies ``is_teacher_correct`` without qualifying
    # the format. We adopt the lenient reading (latent) so the JSONL
    # records the teacher's *semantic* correctness — the format gate
    # ``separator_found`` is already its own field. Downstream filters
    # (``filter_teacher_wrong`` in Phase 2) can still choose which
    # notion of correctness to trust.
    is_teacher_correct = bool(latent_correct or strict_correct)

    # ``total_len_tokens`` under the STUDENT tokenizer (decision 3a).
    # We rebuild the same chat turn structure that ``tokenise_example_cot``
    # will use in Phase 2: system + user + assistant, where assistant
    # content is ``teacher_full_text``. This is the exact input the
    # student training loop will tokenise.
    student_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": teacher_full_text},
    ]
    # return_tensors=None guarantees a plain Python list of ints regardless
    # of transformers version — len() then gives the true token count.
    student_tokenised = student_tokenizer.apply_chat_template(
        student_messages,
        tokenize=True,
        return_tensors=None,
        add_generation_prompt=False,
    )
    total_len_tokens = len(student_tokenised)

    return {
        "split": split,
        "idx": int(idx),
        "question": question,
        "answer_gold": gold if gold is not None else "",
        "teacher_full_text": teacher_full_text,
        "extracted_answer": pred,
        "is_teacher_correct": is_teacher_correct,
        "separator_found": bool(sep_found),
        "total_len_tokens": int(total_len_tokens),
    }


# ------------------------------------------------------------------
# Split enumeration
# ------------------------------------------------------------------

def iter_generation_targets(
    train_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    max_per_split: int | None = None,
) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    """Yield ``(split, idx, example)`` triples in deterministic order.

    Few-shot indices ``FEW_SHOT_INDICES`` are excluded from the train
    pool (decision 1a). Order: train ascending, then test ascending.
    """
    fs = set(FEW_SHOT_INDICES)

    train_yielded = 0
    for i in range(len(train_ds)):
        if i in fs:
            continue
        if max_per_split is not None and train_yielded >= max_per_split:
            break
        yield ("train", i, train_ds[int(i)])
        train_yielded += 1

    test_yielded = 0
    for i in range(len(test_ds)):
        if max_per_split is not None and test_yielded >= max_per_split:
            break
        yield ("test", i, test_ds[int(i)])
        test_yielded += 1


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

def compute_stats(jsonl_path: str) -> Dict[str, Any]:
    """Aggregate stats over the full JSONL, per split and combined."""
    per_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            split = rec.get("split")
            if split in per_split:
                per_split[split].append(rec)

    def _summ(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(records)
        if n == 0:
            return {"n": 0}
        sep = sum(1 for r in records if r.get("separator_found"))
        cor = sum(1 for r in records if r.get("is_teacher_correct"))
        lens = sorted(r["total_len_tokens"] for r in records)
        def _p(p: float) -> float:
            return float(lens[int(round((len(lens) - 1) * p / 100))])
        mean_len = sum(lens) / n
        return {
            "n": n,
            "separator_rate": sep / n,
            "teacher_accuracy": cor / n,
            "mean_total_len_tokens": mean_len,
            "median_total_len_tokens": _p(50),
            "p95_total_len_tokens": _p(95),
            "p99_total_len_tokens": _p(99),
            "max_total_len_tokens": lens[-1],
            "n_over_1024": sum(1 for x in lens if x > 1024),
        }

    out: Dict[str, Any] = {
        "source": jsonl_path,
        "splits": {k: _summ(v) for k, v in per_split.items()},
    }
    all_records = per_split["train"] + per_split["test"]
    out["overall"] = _summ(all_records)
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(
    teacher_name: str = DEFAULT_TEACHER_NAME,
    teacher_load_mode: str = "bf16",
    student_name: str = DEFAULT_STUDENT_NAME,
    output_path: str = DEFAULT_OUTPUT_PATH,
    stats_path: str = DEFAULT_STATS_PATH,
    max_new_tokens: int = 512,
    max_per_split: int | None = None,
    flush_every: int = 20,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ---- Load datasets ----
    print(f"[data] Loading GSM8K train + test …")
    train_ds = datasets.load_dataset("gsm8k", "main", split="train")
    test_ds = datasets.load_dataset("gsm8k", "main", split="test")
    print(f"[data] train={len(train_ds)}  test={len(test_ds)}")

    # ---- Tokenizers ----
    print(f"[tokenizer] teacher={teacher_name}  student={student_name}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    student_tokenizer = AutoTokenizer.from_pretrained(student_name)
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # ---- Resume ----
    done_keys = load_done_keys(output_path)

    # ---- Load teacher (skip if nothing left to do) ----
    remaining_plan = [
        (split, idx, ex)
        for (split, idx, ex) in iter_generation_targets(
            train_ds, test_ds, max_per_split=max_per_split,
        )
        if (split, idx) not in done_keys
    ]
    n_remaining = len(remaining_plan)
    n_target = n_remaining + len(done_keys)
    print(f"[plan] total targets={n_target}  remaining={n_remaining}  "
          f"already_done={len(done_keys)}")

    if n_remaining == 0:
        print("[plan] nothing to generate — writing stats and exiting.")
        stats = compute_stats(output_path)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[stats] saved → {stats_path}")
        return

    print(f"[teacher] Loading {teacher_name} (mode={teacher_load_mode}) …")
    t0 = time.time()
    teacher = load_teacher(teacher_name, teacher_load_mode)
    print(f"[teacher] loaded in {time.time() - t0:.1f}s")
    device = next(teacher.parameters()).device

    # ---- Generation loop ----
    t_loop = time.time()
    n_done_this_run = 0
    with open(output_path, "a", encoding="utf-8") as fout:
        for (split, idx, example) in remaining_plan:
            rec = generate_one(
                split=split,
                idx=idx,
                example=example,
                train_ds=train_ds,
                teacher=teacher,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done_this_run += 1

            if n_done_this_run % flush_every == 0:
                fout.flush()
                elapsed = time.time() - t_loop
                rate = n_done_this_run / max(elapsed, 1e-6)
                eta = (n_remaining - n_done_this_run) / max(rate, 1e-6)
                print(
                    f"[gen] {n_done_this_run:>5d}/{n_remaining}  "
                    f"split={split}  idx={idx:>5d}  "
                    f"sep={rec['separator_found']}  "
                    f"correct={rec['is_teacher_correct']}  "
                    f"len={rec['total_len_tokens']}  "
                    f"({rate:.2f} ex/s, ETA {eta / 60:.1f} min)"
                )
        fout.flush()

    print(f"\n[gen] finished: {n_done_this_run} new example(s) in "
          f"{(time.time() - t_loop) / 60:.1f} min")

    # ---- Stats ----
    stats = compute_stats(output_path)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[stats] saved → {stats_path}")

    # ---- Terminal summary ----
    ov = stats["overall"]
    print("\n" + "=" * 68)
    print("  FULL CoT GENERATION SUMMARY")
    print("=" * 68)
    print(f"  output                  : {output_path}")
    print(f"  n_total                 : {ov['n']}")
    print(f"  separator_rate          : {ov['separator_rate']:.4f}")
    print(f"  teacher_accuracy        : {ov['teacher_accuracy']:.4f}")
    print(f"  mean total_len (student): {ov['mean_total_len_tokens']:.1f}")
    print(f"  p95  total_len (student): {ov['p95_total_len_tokens']:.1f}")
    print(f"  p99  total_len (student): {ov['p99_total_len_tokens']:.1f}")
    print(f"  max  total_len (student): {ov['max_total_len_tokens']}")
    print(f"  n_over_1024 (student)   : {ov['n_over_1024']}")
    for split_name, s in stats["splits"].items():
        if s.get("n", 0) == 0:
            continue
        print(f"  --- {split_name} ---")
        print(f"    n={s['n']}  sep={s['separator_rate']:.4f}  "
              f"acc={s['teacher_accuracy']:.4f}  "
              f"p95={s['p95_total_len_tokens']:.1f}  "
              f"max={s['max_total_len_tokens']}  "
              f"over_1024={s['n_over_1024']}")
    print("=" * 68)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full GSM8K CoT generation (Article II, Phase 1.1).",
    )
    p.add_argument("--teacher-name", default=DEFAULT_TEACHER_NAME)
    p.add_argument("--teacher-load-mode", default="bf16",
                   choices=["bf16", "4bit", "8bit"])
    p.add_argument("--student-name", default=DEFAULT_STUDENT_NAME,
                   help="Used ONLY for measuring total_len_tokens.")
    p.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--stats-path", default=DEFAULT_STATS_PATH)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--max-per-split", type=int, default=None,
        help="Debug / smoke-test cap on train and test each.",
    )
    p.add_argument(
        "--flush-every", type=int, default=20,
        help="Flush JSONL and log progress every N new records.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        teacher_name=args.teacher_name,
        teacher_load_mode=args.teacher_load_mode,
        student_name=args.student_name,
        output_path=args.output_path,
        stats_path=args.stats_path,
        max_new_tokens=args.max_new_tokens,
        max_per_split=args.max_per_split,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
