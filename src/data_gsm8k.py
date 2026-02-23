"""GSM8K data loading, chat-template tokenisation, and collation."""

from __future__ import annotations

from typing import Any, Dict, List

import datasets
import torch
from transformers import PreTrainedTokenizerBase

SYSTEM_PROMPT = "You are a helpful assistant that solves math word problems."


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

    Returns dict with keys ``input_ids``, ``attention_mask``, ``labels``.
    Prompt + answer are both included in the loss (labels = input_ids).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    # apply_chat_template returns a list of token ids.
    # Some tokenizer versions return a tokenizers.Encoding object
    # instead of a plain list — handle both cases.
    raw = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Extract plain list[int] regardless of return type
    if isinstance(raw, list):
        # Could be list[int] or list[Encoding]; flatten safely
        if raw and hasattr(raw[0], "ids"):
            # list of Encoding objects → take ids from them all
            raw_ids: list[int] = []
            for enc in raw:
                raw_ids.extend(enc.ids)
        else:
            raw_ids = raw
    elif hasattr(raw, "ids"):
        # Single Encoding object
        raw_ids = raw.ids
    else:
        raw_ids = list(raw)

    # Truncate to max_length and ensure plain list[int]
    input_ids = [int(x) for x in raw_ids[:max_length]]
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)  # copy – prompt + answer in loss

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ------------------------------------------------------------------
# 4.3  Collator (dynamic padding)
# ------------------------------------------------------------------

class KDCollator:
    """Pad ``input_ids``, ``attention_mask`` (with pad_token_id / 0)
    and ``labels`` (with -100)."""

    def __init__(self, pad_token_id: int, max_length: int):
        assert pad_token_id is not None, (
            "pad_token_id is None – set tokenizer.pad_token = tokenizer.eos_token "
            "before creating the collator."
        )
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Determine the max sequence length in this batch (capped)
        batch_max = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        input_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []

        for f in features:
            seq_len = min(len(f["input_ids"]), batch_max)
            pad_len = batch_max - seq_len

            input_ids_batch.append(f["input_ids"][:seq_len] + [self.pad_token_id] * pad_len)
            attention_mask_batch.append(f["attention_mask"][:seq_len] + [0] * pad_len)
            labels_batch.append(f["labels"][:seq_len] + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }

        # Shape asserts
        B = len(features)
        assert batch["input_ids"].shape == (B, batch_max), (
            f"input_ids shape {batch['input_ids'].shape} != ({B}, {batch_max})"
        )
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape

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
