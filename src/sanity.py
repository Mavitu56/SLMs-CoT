"""
Sanity checks – run on 1 batch before full training.

Each check is a standalone function that raises on failure.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data_gsm8k import build_dataloader
from src.losses_kd import compute_total_loss


# ==================================================================
# Helpers
# ==================================================================

def _get_one_batch(
    tokenizer,
    max_length: int,
    micro_n: int = 4,
    batch_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Return a single batch from GSM8K (train)."""
    loader = build_dataloader(
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        split="train",
        micro_overfit_n=micro_n,
        shuffle=False,
    )
    batch = next(iter(loader))
    return batch


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def _load_small_model(name: str, device: torch.device):
    """Load a small model in bf16 (used for sanity checks)."""
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    return model


# ==================================================================
# 6.1  Teacher is frozen (no gradients)
# ==================================================================

def check_teacher_frozen(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    T: float = 2.0,
) -> None:
    """Forward + backward and verify teacher has no gradients."""
    print("[sanity] 6.1 – Teacher frozen …", end=" ")

    device = next(student.parameters()).device
    batch = _to_device(batch, device)

    # Teacher forward (no_grad)
    teacher.eval()
    with torch.no_grad():
        t_out = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    # Student forward
    s_out = student(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    loss_total, _, _, _ = compute_total_loss(
        teacher_logits=t_out.logits,
        student_logits=s_out.logits,
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
        T=T,
        lambda_kd=1.0,
        lambda_ce=0.5,
    )

    loss_total.backward()

    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher param {name} has grad!"

    # Clean up student grads
    student.zero_grad()
    print("PASSED")


# ==================================================================
# 6.2  Edge-case coefficients
# ==================================================================

def check_lambda_edge_cases(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    T: float = 2.0,
) -> None:
    """lambda_kd=0 → loss == lambda_ce*ce ; lambda_ce=0 → loss == lambda_kd*T²*kd."""
    print("[sanity] 6.2 – Lambda edge cases …", end=" ")

    device = next(student.parameters()).device
    batch = _to_device(batch, device)

    with torch.no_grad():
        t_logits = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        s_logits = student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

    # Case 1: lambda_kd = 0
    loss_total_1, loss_ce_1, _, _ = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, lambda_kd=0.0, lambda_ce=0.5,
    )
    expected_1 = 0.5 * loss_ce_1
    assert torch.allclose(loss_total_1, expected_1, atol=1e-5), (
        f"lambda_kd=0: total={loss_total_1.item():.6f} != 0.5*ce={expected_1.item():.6f}"
    )

    # Case 2: lambda_ce = 0
    loss_total_2, _, loss_kd_2, _ = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, lambda_kd=1.0, lambda_ce=0.0,
    )
    expected_2 = 1.0 * (T ** 2) * loss_kd_2
    assert torch.allclose(loss_total_2, expected_2, atol=1e-5), (
        f"lambda_ce=0: total={loss_total_2.item():.6f} != T²*kd={expected_2.item():.6f}"
    )

    print("PASSED")


# ==================================================================
# 6.3  KD ≈ 0 when teacher == student
# ==================================================================

def check_kd_near_zero_same_model(
    tokenizer,
    max_length: int,
    student_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    T: float = 2.0,
    normal_kd_ref: float | None = None,
) -> None:
    """When both teacher and student are the same model, KD should be ≈ 0.

    Uses a two-tier threshold:
    * Absolute: kd < 1e-2  (generous, covers bf16 noise)
    * Relative (if ``normal_kd_ref`` provided): self-kd must be
      at least 100× smaller than the real teacher-vs-student KD.
    """
    print("[sanity] 6.3 – KD ≈ 0 (teacher == student) …", end=" ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a = _load_small_model(student_name, device)
    model_a.eval()
    model_b = _load_small_model(student_name, device)
    model_b.eval()

    batch = _get_one_batch(tokenizer, max_length, micro_n=2, batch_size=1)
    batch = _to_device(batch, device)

    with torch.no_grad():
        logits_a = model_a(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        logits_b = model_b(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

    _, _, loss_kd, _ = compute_total_loss(
        logits_a, logits_b, batch["labels"], batch["attention_mask"],
        T=T, lambda_kd=1.0, lambda_ce=0.0,
    )

    kd_val = loss_kd.item()

    # Absolute threshold (generous for bf16 / quantisation noise)
    assert kd_val < 1e-2, (
        f"KD loss with identical models should be ≈0, got {kd_val:.6f}"
    )

    # Relative threshold – if we know the normal teacher≠student KD
    if normal_kd_ref is not None and normal_kd_ref > 0:
        ratio = kd_val / normal_kd_ref
        assert ratio < 0.01, (
            f"Self-KD ({kd_val:.2e}) should be ≪ normal KD ({normal_kd_ref:.2e}), "
            f"ratio={ratio:.4f}"
        )
        print(f"PASSED  (kd={kd_val:.2e}, ratio={ratio:.2e})")
    else:
        print(f"PASSED  (kd={kd_val:.2e})")

    # Free memory
    del model_a, model_b
    torch.cuda.empty_cache()


# ==================================================================
# 6.4  KL is non-negative
# ==================================================================

def check_kl_non_negative(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    T: float = 2.0,
) -> None:
    """Forward KL should not be significantly negative."""
    print("[sanity] 6.4 – KL ≥ 0 …", end=" ")

    device = next(student.parameters()).device
    batch = _to_device(batch, device)

    with torch.no_grad():
        t_logits = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        s_logits = student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

    _, _, loss_kd, _ = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, lambda_kd=1.0, lambda_ce=0.0,
    )

    assert loss_kd.item() >= -1e-5, (
        f"KL divergence should be ≥ 0, got {loss_kd.item():.6f}"
    )
    print(f"PASSED  (kd={loss_kd.item():.4f})")


# ==================================================================
# 6.5  Mask works – all labels = -100 → loss = 0
# ==================================================================

def check_mask_all_ignored(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    T: float = 2.0,
) -> None:
    """When all labels are -100, losses should be exactly 0 and no NaN."""
    print("[sanity] 6.5 – Mask (all -100) …", end=" ")

    device = next(student.parameters()).device
    batch = _to_device(batch, device)

    # Clone and mask everything
    batch = {k: v.clone() for k, v in batch.items()}
    batch["labels"][:] = -100

    with torch.no_grad():
        t_logits = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        s_logits = student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

    loss_total, loss_ce, loss_kd, n_valid = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, lambda_kd=1.0, lambda_ce=0.5,
    )

    assert n_valid == 0, f"Expected 0 valid tokens, got {n_valid}"
    assert loss_total.item() == 0.0, f"Expected total=0, got {loss_total.item()}"
    assert not math.isnan(loss_total.item()), "Loss is NaN!"
    assert not math.isnan(loss_ce.item()), "CE is NaN!"
    assert not math.isnan(loss_kd.item()), "KD is NaN!"
    print("PASSED")


# ==================================================================
# 6.6  Padding ↔ attention_mask consistency
# ==================================================================

def check_padding_attention_mask(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    tokenizer,
    max_length: int,
    T: float = 2.0,
) -> None:
    """Collate a batch of 4 examples (different lengths) and verify that
    attention_mask has zeros wherever there is padding, and that
    labels == -100 at those positions."""
    print("[sanity] 6.6 – Padding ↔ attention_mask …", end=" ")

    # Use batch_size > 1 to guarantee padding in at least some rows
    loader = build_dataloader(
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=4,
        split="train",
        micro_overfit_n=8,
        shuffle=False,
    )
    batch = next(iter(loader))
    B, L = batch["input_ids"].shape

    pad_positions = batch["attention_mask"] == 0

    if pad_positions.any():
        # Labels must be -100 where padding
        assert (batch["labels"][pad_positions] == -100).all(), (
            "Labels must be -100 at padded positions."
        )
        # attention_mask zeros should match the right side (left-padded or right-padded)
        n_pad = int(pad_positions.sum().item())
        print(f"PASSED  ({n_pad} pad tokens verified, B={B}, L={L})")
    else:
        # All sequences happen to be the same length – still valid
        print(f"PASSED  (no padding needed, B={B}, L={L})")


# ==================================================================
# Public entry point
# ==================================================================

def run_all_sanity_checks(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    tokenizer,
    cfg: Dict[str, Any],
) -> None:
    """Run all sanity checks. Raises on first failure."""
    T = cfg.get("temperature", 2.0)
    max_length = cfg.get("max_length", 512)
    student_name = cfg.get("student_name", "Qwen/Qwen2.5-0.5B-Instruct")

    batch = _get_one_batch(tokenizer, max_length, micro_n=4, batch_size=1)

    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    check_teacher_frozen(teacher, student, batch, T=T)
    check_lambda_edge_cases(teacher, student, batch, T=T)
    check_kl_non_negative(teacher, student, batch, T=T)
    check_mask_all_ignored(teacher, student, batch, T=T)
    check_padding_attention_mask(teacher, student, tokenizer, max_length, T=T)

    # Compute a "normal" KD reference for the relative threshold in 6.3
    device = next(student.parameters()).device
    _batch_dev = _to_device(batch, device)
    with torch.no_grad():
        _t_logits = teacher(
            input_ids=_batch_dev["input_ids"],
            attention_mask=_batch_dev["attention_mask"],
        ).logits
        _s_logits = student(
            input_ids=_batch_dev["input_ids"],
            attention_mask=_batch_dev["attention_mask"],
        ).logits
    _, _, _normal_kd, _ = compute_total_loss(
        _t_logits, _s_logits, _batch_dev["labels"], _batch_dev["attention_mask"],
        T=T, lambda_kd=1.0, lambda_ce=0.0,
    )
    normal_kd_ref = _normal_kd.item()
    print(f"[info] normal teacher→student KD = {normal_kd_ref:.4f}")

    # 6.3 loads its own models (both copies of student)
    check_kd_near_zero_same_model(
        tokenizer, max_length,
        student_name=student_name, T=T,
        normal_kd_ref=normal_kd_ref,
    )

    print("=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)
