"""Dolly-15k data loading, chat-template tokenisation, and collation.

Dataset: databricks/databricks-dolly-15k
Masking: generic prompt_len approach — labels[:prompt_len] = -100
         so that only the response tokens contribute to the loss.

No region_ids (unlike GSM8K): the only distinction is prompt vs response.
"""

from __future__ import annotations

from typing import Any, Dict, List

import datasets
import torch
from transformers import PreTrainedTokenizerBase

from src.data_gsm8k import KDCollator


# ------------------------------------------------------------------
# Load raw dataset
# ------------------------------------------------------------------

def load_dolly(split: str = "train") -> datasets.Dataset:
    """Load Dolly-15k from Hugging Face.

    The dataset only has a 'train' split. For evaluation, use a
    held-out portion via ``datasets.Dataset.train_test_split()``.
    """
    ds = datasets.load_dataset("databricks/databricks-dolly-15k", split=split)
    return ds


# ------------------------------------------------------------------
# Tokenise a single example using chat template
# ------------------------------------------------------------------

def tokenise_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[int]] | None:
    """Build chat messages and tokenise with the model's chat template.

    Returns dict with keys: ``input_ids``, ``attention_mask``, ``labels``.
    Returns None if the response is empty (caller should filter these out).

    Masking: labels[:prompt_len] = -100 so that only response tokens
    participate in the loss.
    """
    instruction = example["instruction"]
    context = example.get("context", "")
    response = example["response"]

    # Skip examples with empty responses
    if not response or not response.strip():
        return None

    # Build prompt content (instruction + optional context)
    if context and context.strip():
        prompt_content = f"{instruction}\n\n{context}"
    else:
        prompt_content = instruction

    # Step 1: build the full chat-formatted string
    messages = [
        {"role": "user", "content": prompt_content},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Step 2: build the prompt-only text to determine boundary
    prompt_messages = [
        {"role": "user", "content": prompt_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Step 3: tokenise full text
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = [int(x) for x in encoded["input_ids"]]
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    # Step 4: determine prompt boundary
    prompt_encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    prompt_len = len(prompt_encoded["input_ids"])

    # Step 5: mask prompt tokens in labels
    prompt_len = min(prompt_len, len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


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
    """End-to-end: load Dolly → tokenise → DataLoader.

    Logs truncation statistics and empty-response discard rate.
    """
    ds = load_dolly(split)

    if micro_overfit_n is not None and micro_overfit_n > 0:
        ds = ds.select(range(min(micro_overfit_n, len(ds))))

    n_before = len(ds)

    def _tokenise_and_filter(example):
        result = tokenise_example(example, tokenizer, max_length)
        if result is None:
            # Return empty lists — will be filtered out
            return {"input_ids": [], "attention_mask": [], "labels": []}
        return result

    ds = ds.map(
        _tokenise_and_filter,
        remove_columns=ds.column_names,
        writer_batch_size=500,
    )

    # Filter out empty examples (responses that were empty)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    n_after = len(ds)
    n_discarded = n_before - n_after

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
    if n_discarded > 0:
        print(
            f"[data] discarded {n_discarded} examples with empty responses "
            f"({100 * n_discarded / max(n_before, 1):.1f}%)"
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
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader
