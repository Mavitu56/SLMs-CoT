"""
Analytical tests for CoT-specific metrics and JSONL loader (Article II).

These tests use synthetic logits and a synthetic JSONL — no model loading,
no GPU required. They guard the contracts of:

* ``compute_entropy_by_phase``
* ``compute_kl_by_phase`` (implicit, via H_R/H_A consistency)
* ``compute_ece_response_only``
* ``load_gsm8k_cot_jsonl`` (filter_no_separator semantics)
* ``tokenise_example_cot`` (region partition sums to seq_len)

Reference
---------
* Plano §2.6 — Tarefa 2.6 (Phase 2, Article II).
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from src.evaluation.analysis_metrics import (
    REGION_PROMPT_ID,
    REGION_REASONING_ID,
    REGION_ANSWER_ID,
    compute_entropy_by_phase,
    compute_ece_response_only,
)


# ==================================================================
# Test 1 — Entropy-by-phase consistency
# ==================================================================

def test_entropy_by_phase_consistency():
    """H_total should equal the token-weighted average of H_R and H_A.

    The function defines H_total over (REASONING ∪ ANSWER); prompt
    tokens are excluded by construction. With known counts n_R and n_A,
    the relation H_total · (n_R + n_A) ≈ n_R · H_R + n_A · H_A must hold
    exactly (modulo float precision).
    """
    torch.manual_seed(0)
    B, L, V = 2, 10, 100

    logits = torch.randn(B, L, V)

    # region_ids: 4 prompt, 3 reasoning, 3 answer (per row)
    region_ids = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]] * B,
        dtype=torch.long,
    )
    valid_mask = torch.ones(B, L, dtype=torch.bool)
    # Prompt tokens are not "valid" for loss in real usage; here we keep
    # them valid in the mask but the function should still ignore them
    # because they belong to neither REASONING nor ANSWER.

    out = compute_entropy_by_phase(logits, region_ids, valid_mask)

    n_R = out["n_R"]
    n_A = out["n_A"]
    assert n_R == 6, f"expected 6 reasoning tokens, got {n_R}"
    assert n_A == 6, f"expected 6 answer tokens, got {n_A}"

    weighted = (n_R * out["H_R"] + n_A * out["H_A"]) / (n_R + n_A)
    assert abs(weighted - out["H_total"]) < 1e-4, (
        f"H_total ({out['H_total']:.6f}) != weighted avg ({weighted:.6f})"
    )

    # ρ = H_R / H_A must be finite when both phases have tokens
    assert out["rho"] == out["rho"], "rho should not be NaN"
    assert out["rho"] > 0


# ==================================================================
# Test 2 — Region partition sanity
# ==================================================================

def test_region_partition_sanity():
    """For a tokenised CoT example, region_ids must partition seq_len.

    Each token gets exactly one region label; the three counts must
    sum to len(input_ids).
    """
    from transformers import AutoTokenizer
    from src.data.data_gsm8k import (
        tokenise_example_cot,
        REGION_PROMPT,
        REGION_REASONING,
        REGION_ANSWER,
    )

    pytest.importorskip("transformers")

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as e:
        pytest.skip(f"Tokenizer download unavailable: {e}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    example = {
        "question": "What is 2 + 3?",
        "teacher_full_text": "Two plus three equals five.\n#### 5",
    }

    result = tokenise_example_cot(example, tokenizer, max_length=256)
    region_ids = result["region_ids"]
    seq_len = len(result["input_ids"])

    n_p = sum(1 for r in region_ids if r == REGION_PROMPT)
    n_r = sum(1 for r in region_ids if r == REGION_REASONING)
    n_a = sum(1 for r in region_ids if r == REGION_ANSWER)

    assert n_p + n_r + n_a == seq_len, (
        f"region partition mismatch: prompt={n_p}, "
        f"reasoning={n_r}, answer={n_a}, seq_len={seq_len}"
    )
    assert n_p > 0, "expected at least one prompt token"
    assert n_r > 0, "expected at least one reasoning token"
    assert n_a > 0, "expected at least one answer token (####)"


# ==================================================================
# Test 3 — ECE_response excludes reasoning tokens
# ==================================================================

def test_ece_response_excludes_reasoning():
    """ECE_response must depend only on REGION_ANSWER tokens.

    Strategy: build two logit tensors that agree exactly on ANSWER
    positions but differ wildly on REASONING positions. Their
    ECE_response values must be identical.
    """
    torch.manual_seed(0)
    B, L, V = 1, 20, 50

    region_ids = torch.zeros(B, L, dtype=torch.long)
    region_ids[0, :8] = REGION_PROMPT_ID
    region_ids[0, 8:14] = REGION_REASONING_ID
    region_ids[0, 14:] = REGION_ANSWER_ID

    labels = torch.randint(0, V, (B, L))
    valid_mask = torch.ones(B, L, dtype=torch.bool)

    base_logits = torch.randn(B, L, V)

    # Variant: identical on ANSWER, perturbed on REASONING
    perturbed = base_logits.clone()
    perturbed[0, 8:14, :] = perturbed[0, 8:14, :] + 10.0 * torch.randn(6, V)

    ece_a = compute_ece_response_only(base_logits, labels, region_ids, valid_mask)
    ece_b = compute_ece_response_only(perturbed, labels, region_ids, valid_mask)

    assert ece_a["n_A"] == ece_b["n_A"] == 6
    assert abs(ece_a["ECE_response"] - ece_b["ECE_response"]) < 1e-6, (
        f"ECE_response should be invariant to reasoning logits: "
        f"{ece_a['ECE_response']:.6e} vs {ece_b['ECE_response']:.6e}"
    )


# ==================================================================
# Test 4 — JSONL loading filters
# ==================================================================

def test_jsonl_loading_filters():
    """load_gsm8k_cot_jsonl must apply filter_no_separator correctly."""
    from src.data.data_gsm8k import load_gsm8k_cot_jsonl

    records = []
    for i in range(8):
        records.append({
            "split": "train",
            "idx": i,
            "question": f"q{i}",
            "answer_gold": "5",
            "teacher_full_text": "answer is 5\n#### 5",
            "extracted_answer": "5",
            "is_teacher_correct": True,
            "separator_found": True,
            "total_len_tokens": 100,
        })
    # 2 records without separator
    for i in range(8, 10):
        records.append({
            "split": "train",
            "idx": i,
            "question": f"q{i}",
            "answer_gold": "5",
            "teacher_full_text": "answer is 5 (no separator here)",
            "extracted_answer": None,
            "is_teacher_correct": False,
            "separator_found": False,
            "total_len_tokens": 100,
        })

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        kept_filtered = load_gsm8k_cot_jsonl(
            path, split="train", filter_no_separator=True,
        )
        kept_unfiltered = load_gsm8k_cot_jsonl(
            path, split="train", filter_no_separator=False,
        )

    assert len(kept_filtered) == 8, (
        f"filter_no_separator=True should keep 8, got {len(kept_filtered)}"
    )
    assert len(kept_unfiltered) == 10, (
        f"filter_no_separator=False should keep 10, got {len(kept_unfiltered)}"
    )
    assert all(r["separator_found"] for r in kept_filtered)
