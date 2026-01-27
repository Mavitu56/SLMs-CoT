from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from datasets import Dataset as HFDataset, load_dataset


def _gsm8k_final_answer(answer: str) -> str:
    """Extract GSM8K final numeric answer.

    Scientific validity fix (confounding control): for the traditional baseline
    we treat the *completion* as the final answer only (no rationale), so that
    the reasoning-aware condition differs by the presence of teacher reasoning.
    """

    if not answer:
        return ""
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return str(answer).strip()


def _maybe_limit(ds: HFDataset, limit: Optional[int]) -> HFDataset:
    if limit is None or limit <= 0:
        return ds
    return ds.select(range(min(len(ds), int(limit))))


def load_gsm8k(split: str = "train", limit: Optional[int] = None, seed: int = 42) -> HFDataset:
    # Try multiple dataset sources for GSM8K (original was deprecated/moved)
    gsm8k_sources = [
        ("openai/gsm8k", "main"),      # New official location
        ("gsm8k", "main"),              # Original (may be unavailable)
    ]
    ds = None
    last_error = None
    for repo_id, config in gsm8k_sources:
        try:
            ds = load_dataset(repo_id, config, split=split, trust_remote_code=True)
            break
        except Exception as e:
            last_error = e
            continue
    if ds is None:
        raise RuntimeError(
            f"Não foi possível carregar o GSM8K de nenhuma fonte conhecida. "
            f"Último erro: {last_error}"
        )
    ds = _maybe_limit(ds, limit)

    def format_ex(example: Dict[str, Any]) -> Dict[str, Any]:
        q = str(example.get("question", ""))
        a_full = str(example.get("answer", ""))
        a_final = _gsm8k_final_answer(a_full)
        prompt = f"Q: {q}\nA:"
        completion = f" {a_final}" if a_final else ""
        return {
            # `text` is the model input sequence for training.
            "text": prompt + completion,
            "prompt": prompt,
            "completion": completion,
            "question": q,
            "answer": a_full,
            "final_answer": a_final,
            "domain": "gsm8k",
        }

    return ds.map(format_ex)


def load_bbh_subset(limit: Optional[int] = None, seed: int = 42) -> HFDataset:
    """Loads a BBH-like logical task via BBEH repo direct download logic (kept minimal)."""

    # Reuse the same task set as evaluation to avoid scope changes.
    # Scientific validity fix (leakage): use the *train* split produced by a
    # deterministic hash-based partition; evaluation uses the complementary split.
    from eval import load_bbeh_task_dataset, split_bbeh_dataset

    task = "logical_deduction_five_objects"
    ds, _src = load_bbeh_task_dataset(task)
    ds_train, _ds_eval = split_bbeh_dataset(ds, seed=seed, eval_fraction=0.2)
    ds = ds_train
    ds = _maybe_limit(ds, limit)

    def format_ex(example: Dict[str, Any]) -> Dict[str, Any]:
        question = str(example.get("input") or example.get("prompt") or "")
        answer = str(example.get("target") or example.get("output") or "")
        prompt = f"{question}\nAnswer:"
        completion = f" {answer}" if answer else ""
        return {
            "text": prompt + completion,
            "prompt": prompt,
            "completion": completion,
            "question": question,
            "answer": answer,
            "domain": "bbh",
        }

    return ds.map(format_ex)


def load_training_dataset(
    enable_gsm8k: bool,
    enable_bbh: bool,
    train_limit: Optional[int],
    seed: int,
) -> HFDataset:
    # Minimal: training uses whichever datasets are enabled. If both enabled, concatenate.
    from datasets import concatenate_datasets

    parts = []
    if enable_gsm8k:
        parts.append(load_gsm8k(split="train", limit=train_limit, seed=seed))
    if enable_bbh:
        # BBH has no train split here; we keep it small and deterministic.
        parts.append(load_bbh_subset(limit=min(100, train_limit or 100), seed=seed))

    if not parts:
        raise ValueError("Nenhum dataset de treino ativado. Ative gsm8k e/ou bbh.")

    if len(parts) == 1:
        return parts[0]

    return concatenate_datasets(parts)
