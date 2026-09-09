"""Unit tests for ScienceQA multimodal dataset, collator, answer extraction, and vocab alignment.

No GPU required — tests data contracts, collator invariance, and answer parsing.
"""

from __future__ import annotations

import torch

from src.data.data_scienceqa import (
    MultimodalKDCollator,
    answer_index_to_letter,
    answer_letter_to_index,
    format_choices,
    REGION_PROMPT,
    REGION_REASONING,
    REGION_ANSWER,
)
from src.evaluation.evaluate_scienceqa import extract_scienceqa_answer
from src.losses.losses_kd import compute_total_loss
from src.training.sanity import _align_vocab


# ==================================================================
# Test 1 — ScienceQA helper functions
# ==================================================================

def test_answer_index_letter_conversion():
    assert answer_index_to_letter(0) == "A"
    assert answer_index_to_letter(1) == "B"
    assert answer_index_to_letter(2) == "C"
    assert answer_index_to_letter(3) == "D"
    assert answer_index_to_letter(4) == "E"

    assert answer_letter_to_index("A") == 0
    assert answer_letter_to_index("b") == 1
    assert answer_letter_to_index(" C ") == 2


def test_format_choices():
    choices = ["Mars", "Venus", "Earth"]
    formatted = format_choices(choices)
    assert "(A) Mars" in formatted
    assert "(B) Venus" in formatted
    assert "(C) Earth" in formatted


# ==================================================================
# Test 2 — Answer extraction regex
# ==================================================================

def test_extract_scienceqa_answer():
    # Primary format
    assert extract_scienceqa_answer("Step 1... #### B") == "B"
    assert extract_scienceqa_answer("Therefore ####  c") == "C"

    # Secondary formats
    assert extract_scienceqa_answer("The answer is (A).") == "A"
    assert extract_scienceqa_answer("the answer is D") == "D"
    assert extract_scienceqa_answer("Answer: (B)") == "B"
    assert extract_scienceqa_answer("Answer: E") == "E"
    assert extract_scienceqa_answer(r"Final answer: \boxed{C}") == "C"

    # Fallback standalone
    assert extract_scienceqa_answer("So we pick option B") == "B"


# ==================================================================
# Test 3 — MultimodalKDCollator
# ==================================================================

def test_multimodal_kd_collator_mixed_batch():
    pad_id = 0
    collator = MultimodalKDCollator(pad_token_id=pad_id, max_length=64)

    # Example 1: with image
    ex1 = {
        "input_ids": torch.tensor([10, 20, 30, 40, 50]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "labels": torch.tensor([-100, -100, 30, 40, 50]),
        "region_ids": torch.tensor([0, 0, 1, 1, 2]),
        "has_image": True,
        "pixel_values": torch.randn(8, 64),
        "image_grid_thw": torch.tensor([[1, 2, 4]]),
    }

    # Example 2: text only (no image)
    ex2 = {
        "input_ids": torch.tensor([10, 20, 35]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([-100, 20, 35]),
        "region_ids": torch.tensor([0, 1, 2]),
        "has_image": False,
    }

    batch = collator([ex1, ex2])

    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["region_ids"].shape == (2, 5)

    # Padding invariants
    assert batch["attention_mask"][1, 3:].tolist() == [0, 0]
    assert batch["labels"][1, 3:].tolist() == [-100, -100]
    assert batch["region_ids"][1, 3:].tolist() == [-1, -1]

    # Visual tensors
    assert "pixel_values" in batch
    assert "image_grid_thw" in batch
    assert batch["pixel_values"].shape == (8, 64)
    assert batch["image_grid_thw"].shape == (1, 3)
    assert batch["has_image"].tolist() == [True, False]


# ==================================================================
# Test 4 — Vocab alignment (7B vs 3B hardware padding mismatch)
# ==================================================================

def test_vocab_alignment_teacher_student():
    """Verify that 152064 -> 151936 truncation preserves loss computation."""
    B, L = 2, 8
    V_teacher = 152064
    V_student = 151936

    t_logits = torch.randn(B, L, V_teacher)
    s_logits = torch.randn(B, L, V_student, requires_grad=True)

    aligned_t, aligned_s = _align_vocab(t_logits, s_logits)

    assert aligned_t.shape == (B, L, V_student)
    assert aligned_s.shape == (B, L, V_student)

    labels = torch.randint(0, V_student, (B, L))
    labels[:, :3] = -100
    attention_mask = torch.ones(B, L, dtype=torch.long)

    loss_total, loss_ce, loss_kd, n_valid = compute_total_loss(
        teacher_logits=aligned_t,
        student_logits=aligned_s,
        labels=labels,
        attention_mask=attention_mask,
        T=2.0,
        alpha=0.5,
        kd_mode="fkl",
    )

    assert torch.isfinite(loss_total)
    assert n_valid > 0
    loss_total.backward()
    assert s_logits.grad is not None
