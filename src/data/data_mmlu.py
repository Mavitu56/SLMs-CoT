"""MMLU data loading, chat-template tokenisation, and collation.

Dataset: cais/mmlu (test split)
Purpose: secondary evaluation only (not training).

Each example is a multiple-choice question with 4 options (A/B/C/D).
The model is prompted with the question and expected to output the
correct letter. Masking uses the generic prompt_len approach.
"""

from __future__ import annotations

from typing import Any, Dict, List

import datasets
import torch
from transformers import PreTrainedTokenizerBase

from src.data.data_gsm8k import KDCollator


# ------------------------------------------------------------------
# Load raw dataset
# ------------------------------------------------------------------

def load_mmlu(
    split: str = "test",
    subject: str | None = None,
) -> datasets.Dataset:
    """Load MMLU from Hugging Face (cais/mmlu).

    Parameters
    ----------
    split : str
        Dataset split ('test', 'validation', 'dev', etc.).
    subject : str or None
        If provided, load only that subject (e.g. 'abstract_algebra').
        If None, loads 'all'.
    """
    config_name = subject if subject else "all"
    ds = datasets.load_dataset("cais/mmlu", config_name, split=split)
    return ds


# ------------------------------------------------------------------
# Format a single MMLU example
# ------------------------------------------------------------------

_CHOICE_LETTERS = "ABCD"


def format_mmlu_prompt(example: Dict[str, Any]) -> tuple[str, str]:
    """Format an MMLU example into (prompt_text, answer_letter).

    Returns
    -------
    prompt_text : str
        The question with choices, ending with "Answer:"
    answer_letter : str
        The correct letter (A/B/C/D)
    """
    question = example["question"]
    choices = example["choices"]

    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{_CHOICE_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer:")

    prompt_text = "\n".join(lines)
    answer_letter = _CHOICE_LETTERS[example["answer"]]

    return prompt_text, answer_letter


# ------------------------------------------------------------------
# Tokenise a single example using chat template
# ------------------------------------------------------------------

def tokenise_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[int]]:
    """Build chat messages and tokenise with the model's chat template.

    Returns dict with keys: ``input_ids``, ``attention_mask``, ``labels``.

    Masking: labels[:prompt_len] = -100 so only the answer token(s)
    participate in the loss/evaluation.
    """
    prompt_text, answer_letter = format_mmlu_prompt(example)

    # Step 1: build the full chat-formatted string
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": f" {answer_letter}"},
    ]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Step 2: build the prompt-only text to determine boundary
    prompt_messages = [
        {"role": "user", "content": prompt_text},
    ]
    prompt_only_text = tokenizer.apply_chat_template(
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
        prompt_only_text,
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
    split: str = "test",
    subject: str | None = None,
    micro_overfit_n: int | None = None,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """End-to-end: load MMLU → tokenise → DataLoader.

    Default: no shuffle (evaluation).
    """
    ds = load_mmlu(split, subject=subject)

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
        f"[data] mmlu {split}: {n_total} examples, "
        f"avg_len={avg_len:.0f}, "
        f"truncated={n_truncated}/{n_total} "
        f"({100 * n_truncated / max(n_total, 1):.1f}%) "
        f"at max_length={max_length}"
    )

    # NOTE: do NOT call ds.set_format("torch") — the HF TorchFormatter imports
    # VideoReader from torchvision.io, unavailable in some Colab torch/torchvision
    # combos. KDCollator tensorises plain Python lists, so default format works.

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
