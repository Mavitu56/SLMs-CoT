"""GSM8K data loading, chat-template tokenisation, and collation.

Region Segmentation
-------------------
Each token receives a **region_id** that identifies its role:

    0 = prompt   (system + user turns, including special tokens)
    1 = reasoning (assistant content before ``####``)
    2 = answer    (``####`` and final numeric answer)

This allows analysis code to compute metrics per region without
contaminating reasoning/answer statistics with prompt tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List

import datasets
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

    ds.set_format("torch")

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
