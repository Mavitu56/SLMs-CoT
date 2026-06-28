"""GSM8K data loading, chat-template tokenisation, and collation.

Region Segmentation
-------------------
Each token receives a **region_id** that identifies its role:

    0 = prompt   (system + user turns, including special tokens)
    1 = reasoning (assistant content before ``####``)
    2 = answer    (``####`` and final numeric answer)

This allows analysis code to compute metrics per region without
contaminating reasoning/answer statistics with prompt tokens.

Two data sources are supported:

* ``load_gsm8k`` + ``tokenise_example`` + ``build_dataloader`` —
  raw GSM8K with the dataset's native ``answer`` field as the
  assistant turn (Article I-style).

* ``load_gsm8k_cot_jsonl`` + ``tokenise_example_cot`` +
  ``build_dataloader_cot`` — Phase 1.1 JSONL produced by the teacher
  (Qwen2.5-7B-Instruct, greedy bf16) with the teacher-generated CoT
  as the assistant turn. This is the off-policy supervision signal
  used for the FKL/RKL comparison in Article II.

References
----------
* Hsieh et al. 2023 — *Distilling Step-by-Step* — teacher-generated
  rationales as supervision signal for CoT-KD.
* Gu et al. 2024 — *MiniLLM* (ICLR) — off-policy KD over fixed
  teacher outputs; consistency with ``P_teacher`` is critical for
  the FKL vs RKL comparison.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

SYSTEM_PROMPT = "You are a helpful assistant that solves math word problems."

# Region IDs (kept as module constants for external reference)
REGION_PROMPT    = 0
REGION_REASONING = 1
REGION_ANSWER    = 2


# ------------------------------------------------------------------
# 4.1  Load raw dataset
# ------------------------------------------------------------------

def load_gsm8k(split: str = "train") -> datasets.Dataset:
    """Load GSM8K (main config) from Hugging Face."""
    ds = datasets.load_dataset("gsm8k", "main", split=split)
    return ds


# ------------------------------------------------------------------
# 4.2  Tokenise a single example using chat template
# ------------------------------------------------------------------

def tokenise_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[int]]:
    """Build chat messages and tokenise with the model's chat template.

    Returns dict with keys:
        ``input_ids``, ``attention_mask``, ``labels``, ``region_ids``.

    ``region_ids`` encodes the role of each token:
        0 = prompt, 1 = reasoning, 2 = answer (after ``####``).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    # Step 1: build the full chat-formatted string
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Step 2: build the prompt-only text (system + user + generation prompt)
    # so we can determine the boundary between prompt and assistant content.
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,  # includes "<|im_start|>assistant\n"
    )

    # Step 3: tokenise full text
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,   # chat template already added them
    )

    input_ids = [int(x) for x in encoded["input_ids"]]
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)  # copy – prompt + answer in loss

    # Step 4: determine prompt boundary (number of prompt tokens)
    prompt_encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    prompt_len = len(prompt_encoded["input_ids"])

    # Step 5: determine #### boundary within the full text
    hash_marker = "####"
    hash_pos_in_text = text.find(hash_marker)
    if hash_pos_in_text >= 0:
        pre_hash_text = text[:hash_pos_in_text]
        pre_hash_encoded = tokenizer(
            pre_hash_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        answer_start = len(pre_hash_encoded["input_ids"])
    else:
        # No #### found (shouldn't happen in GSM8K, but be safe)
        answer_start = len(input_ids)

    # Step 6: build region_ids
    seq_len = len(input_ids)
    region_ids: List[int] = []
    for i in range(seq_len):
        if i < prompt_len:
            region_ids.append(REGION_PROMPT)
        elif i < answer_start:
            region_ids.append(REGION_REASONING)
        else:
            region_ids.append(REGION_ANSWER)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "region_ids": region_ids,
    }


# ------------------------------------------------------------------
# 4.3  Collator (dynamic padding)
# ------------------------------------------------------------------

class KDCollator:
    """Pad ``input_ids``, ``attention_mask`` (with pad_token_id / 0),
    ``labels`` (with -100), and ``region_ids`` (with -1)."""

    def __init__(self, pad_token_id: int, max_length: int):
        assert pad_token_id is not None, (
            "pad_token_id is None – set tokenizer.pad_token = tokenizer.eos_token "
            "before creating the collator."
        )
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Ensure plain lists (set_format("torch") may yield tensors)
        _keys = ["input_ids", "attention_mask", "labels"]
        if "region_ids" in features[0]:
            _keys.append("region_ids")
        for f in features:
            for k in _keys:
                if isinstance(f[k], torch.Tensor):
                    f[k] = f[k].tolist()

        has_regions = "region_ids" in features[0]

        # Determine the max sequence length in this batch (capped)
        batch_max = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        input_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        region_ids_batch: List[List[int]] = []

        for f in features:
            seq_len = min(len(f["input_ids"]), batch_max)
            pad_len = batch_max - seq_len

            input_ids_batch.append(f["input_ids"][:seq_len] + [self.pad_token_id] * pad_len)
            attention_mask_batch.append(f["attention_mask"][:seq_len] + [0] * pad_len)
            labels_batch.append(f["labels"][:seq_len] + [-100] * pad_len)
            if has_regions:
                region_ids_batch.append(f["region_ids"][:seq_len] + [-1] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
        if has_regions:
            batch["region_ids"] = torch.tensor(region_ids_batch, dtype=torch.long)

        # Shape asserts
        B = len(features)
        assert batch["input_ids"].shape == (B, batch_max), (
            f"input_ids shape {batch['input_ids'].shape} != ({B}, {batch_max})"
        )
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        if has_regions:
            assert batch["region_ids"].shape == batch["input_ids"].shape

        # Padding-consistency assert: where attention_mask==0, labels must be -100
        pad_positions = batch["attention_mask"] == 0
        if pad_positions.any():
            assert (batch["labels"][pad_positions] == -100).all(), (
                "Labels must be -100 at every padded position (attention_mask==0)."
            )

        return batch


# ------------------------------------------------------------------
# Convenience: build tokenised dataset + dataloader
# ------------------------------------------------------------------

def build_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    split: str = "train",
    micro_overfit_n: int | None = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """End-to-end: load GSM8K → tokenise → DataLoader.

    Logs truncation statistics so the user can judge whether
    ``max_length`` is large enough for the dataset.
    """
    ds = load_gsm8k(split)

    if micro_overfit_n is not None and micro_overfit_n > 0:
        ds = ds.select(range(min(micro_overfit_n, len(ds))))

    ds = ds.map(
        lambda ex: tokenise_example(ex, tokenizer, max_length),
        remove_columns=ds.column_names,
        writer_batch_size=500,
    )

    # ---- Truncation statistics ----
    lengths = [len(row["input_ids"]) for row in ds]
    n_total = len(lengths)
    n_truncated = sum(1 for l in lengths if l == max_length)
    avg_len = sum(lengths) / max(n_total, 1)
    print(
        f"[data] {split}: {n_total} examples, "
        f"avg_len={avg_len:.0f}, "
        f"truncated={n_truncated}/{n_total} "
        f"({100 * n_truncated / max(n_total, 1):.1f}%) "
        f"at max_length={max_length}"
    )

    # NOTE: do NOT call ds.set_format("torch"). The HF TorchFormatter imports
    # VideoReader from torchvision.io, which is unavailable in some Colab
    # torch/torchvision combos and raises ImportError inside the DataLoader
    # worker. KDCollator already tensorises plain Python lists, so the default
    # (list) format is both sufficient and portable.

    collator = KDCollator(
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
        max_length=max_length,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        drop_last=False,
    )
    return loader


# ------------------------------------------------------------------
# CoT JSONL pathway (Phase 1.1 output)
# ------------------------------------------------------------------

def load_gsm8k_cot_jsonl(
    jsonl_path: str,
    split: str = "train",
    filter_no_separator: bool = True,
    filter_teacher_wrong: bool = False,
) -> List[Dict[str, Any]]:
    """Load teacher-generated CoT dataset from JSONL (Phase 1.1 output).

    Each JSONL line is expected to contain the schema produced by
    ``scripts/generate_full_cot.py``:

        split, idx, question, answer_gold, teacher_full_text,
        extracted_answer, is_teacher_correct, separator_found,
        total_len_tokens.

    Reference
    ---------
    Hsieh et al. 2023 (*Distilling Step-by-Step*) — teacher-generated
    rationales as supervision signal for student training.

    Parameters
    ----------
    jsonl_path : str
        Path to the JSONL file (e.g. ``data/gsm8k_cot_qwen25_7b.jsonl``).
    split : str
        ``'train'`` or ``'test'`` — selects records by their ``split`` field.
    filter_no_separator : bool
        If True, drop records where ``separator_found == False`` (default).
        These records have an empty ANSWER region after tokenisation
        and would dilute per-region metrics.
    filter_teacher_wrong : bool
        Decision D3 (resolved off, 2026-04-24). If True, drop records
        where the teacher's extracted answer disagrees with the gold.

    Returns
    -------
    List of dicts ready to be tokenised by ``tokenise_example_cot``.
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(
            f"CoT JSONL not found: {jsonl_path!r}. "
            f"Run scripts/generate_full_cot.py first."
        )

    if split not in ("train", "test"):
        raise ValueError(f"Invalid split: {split!r}. Use 'train' or 'test'.")

    n_total = 0
    n_wrong_split = 0
    n_dropped_separator = 0
    n_dropped_teacher_wrong = 0
    records: List[Dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if r.get("split") != split:
                n_wrong_split += 1
                continue
            if filter_no_separator and not r.get("separator_found", False):
                n_dropped_separator += 1
                continue
            if filter_teacher_wrong and not r.get("is_teacher_correct", False):
                n_dropped_teacher_wrong += 1
                continue
            records.append(r)

    print(
        f"[data] CoT JSONL split={split}: {len(records)} kept "
        f"(total_lines={n_total}, other_split={n_wrong_split}, "
        f"dropped_no_sep={n_dropped_separator}, "
        f"dropped_teacher_wrong={n_dropped_teacher_wrong})"
    )
    return records


def tokenise_example_cot(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[int]]:
    """Tokenise a GSM8K-CoT example using the teacher-generated rationale.

    Identical to ``tokenise_example`` except that the assistant turn is
    populated from ``example["teacher_full_text"]`` instead of the native
    GSM8K ``answer`` field. The 3-region segmentation is preserved
    (``REGION_PROMPT=0``, ``REGION_REASONING=1``, ``REGION_ANSWER=2``)
    via ``text.find('####')``.

    Reference
    ---------
    Gu et al. 2024 (*MiniLLM*, ICLR) — off-policy KD consumes fixed
    teacher outputs; consistency with the teacher distribution is
    critical for the FKL vs RKL comparison.

    Parameters
    ----------
    example : dict
        A record from ``load_gsm8k_cot_jsonl``. Required keys:
        ``question``, ``teacher_full_text``.
    tokenizer : PreTrainedTokenizerBase
    max_length : int

    Returns
    -------
    dict with keys ``input_ids``, ``attention_mask``, ``labels``,
    ``region_ids``.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["teacher_full_text"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = [int(x) for x in encoded["input_ids"]]
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    prompt_encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    prompt_len = len(prompt_encoded["input_ids"])

    # Mask prompt tokens — only reasoning + answer enter the loss
    prompt_len_eff = min(prompt_len, len(labels))
    labels[:prompt_len_eff] = [-100] * prompt_len_eff

    # Locate '####' boundary inside the full text
    hash_marker = "####"
    hash_pos_in_text = text.find(hash_marker)
    if hash_pos_in_text >= 0:
        pre_hash_text = text[:hash_pos_in_text]
        pre_hash_encoded = tokenizer(
            pre_hash_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        answer_start = len(pre_hash_encoded["input_ids"])
    else:
        answer_start = len(input_ids)

    seq_len = len(input_ids)
    region_ids: List[int] = []
    for i in range(seq_len):
        if i < prompt_len_eff:
            region_ids.append(REGION_PROMPT)
        elif i < answer_start:
            region_ids.append(REGION_REASONING)
        else:
            region_ids.append(REGION_ANSWER)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "region_ids": region_ids,
    }


def build_dataloader_cot(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    jsonl_path: str,
    split: str = "train",
    filter_teacher_wrong: bool = False,
    micro_overfit_n: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """End-to-end CoT pathway: load JSONL → tokenise → DataLoader.

    Reuses the existing ``KDCollator`` (already supports ``region_ids``
    with padding ``-1``). Reproducibility is ensured via
    ``torch.Generator(seed)`` and a numpy/random ``worker_init_fn``,
    matching the pattern in ``data_dolly.py``.

    Reference
    ---------
    Hsieh et al. 2023 — *Distilling Step-by-Step*.
    Gu et al. 2024 — *MiniLLM* (ICLR), off-policy KD pipeline.
    """
    records = load_gsm8k_cot_jsonl(
        jsonl_path=jsonl_path,
        split=split,
        filter_no_separator=True,
        filter_teacher_wrong=filter_teacher_wrong,
    )

    if micro_overfit_n is not None and micro_overfit_n > 0:
        records = records[: min(micro_overfit_n, len(records))]

    ds = datasets.Dataset.from_list(records)
    ds = ds.map(
        lambda ex: tokenise_example_cot(ex, tokenizer, max_length),
        remove_columns=ds.column_names,
        writer_batch_size=500,
    )

    # Truncation statistics
    lengths = [len(row["input_ids"]) for row in ds]
    n_total = len(lengths)
    n_truncated = sum(1 for l in lengths if l == max_length)
    avg_len = sum(lengths) / max(n_total, 1)
    print(
        f"[data] CoT {split}: {n_total} examples, "
        f"avg_len={avg_len:.0f}, "
        f"truncated={n_truncated}/{n_total} "
        f"({100 * n_truncated / max(n_total, 1):.1f}%) "
        f"at max_length={max_length}"
    )

    # NOTE: do NOT call ds.set_format("torch") — see build_dataloader() above.
    # KDCollator tensorises plain Python lists; the HF TorchFormatter's
    # VideoReader import breaks on some Colab torch/torchvision combos.

    collator = KDCollator(
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
        max_length=max_length,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init_fn(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=g if shuffle else None,
        worker_init_fn=_worker_init_fn,
    )
    return loader
