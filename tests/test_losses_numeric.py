"""Analytical numeric tests for KD losses (Fase 3).

All tests use small synthetic tensors on CPU — no model loading required.
"""

import torch
import pytest

from src.losses.losses_kd import (
    compute_kd_forward_kl,
    compute_kd_reverse_kl,
    compute_total_loss,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_toy_data(B=2, L=10, V=100, seed=42):
    """Create toy logits, labels, and attention mask for compute_total_loss."""
    g = torch.Generator().manual_seed(seed)
    teacher_logits = torch.randn(B, L, V, generator=g)
    student_logits = torch.randn(B, L, V, generator=g)
    labels = torch.randint(0, V, (B, L), generator=g)
    labels[:, :3] = -100  # mask first 3 positions as prompt
    attention_mask = torch.ones(B, L, dtype=torch.long)
    return teacher_logits, student_logits, labels, attention_mask


# ------------------------------------------------------------------
# Test 1: KL(p || p) = 0 for both FKL and RKL
# ------------------------------------------------------------------

def test_kl_self_is_zero():
    """When teacher == student, both FKL and RKL should be ~0."""
    torch.manual_seed(42)
    logits = torch.randn(2, 5, 100)
    mask = torch.ones(2, 5, dtype=torch.bool)

    loss_fkl = compute_kd_forward_kl(logits, logits, mask, T=1.0)
    assert loss_fkl.item() < 1e-5, f"FKL self should be ~0, got {loss_fkl.item()}"

    loss_rkl = compute_kd_reverse_kl(logits, logits, mask)
    assert loss_rkl.item() < 1e-5, f"RKL self should be ~0, got {loss_rkl.item()}"


# ------------------------------------------------------------------
# Test 2: KL(p || q) > 0 for p != q
# ------------------------------------------------------------------

def test_kl_different_is_positive():
    """KL divergence between different distributions must be > 0."""
    torch.manual_seed(42)
    t_logits = torch.randn(2, 5, 100)
    s_logits = torch.randn(2, 5, 100)
    mask = torch.ones(2, 5, dtype=torch.bool)

    fkl = compute_kd_forward_kl(t_logits, s_logits, mask, T=1.0)
    assert fkl.item() > 0, f"FKL should be > 0, got {fkl.item()}"

    rkl = compute_kd_reverse_kl(s_logits, t_logits, mask)
    assert rkl.item() > 0, f"RKL should be > 0, got {rkl.item()}"


# ------------------------------------------------------------------
# Test 3: Normalisation — loss invariant to batch duplication
# ------------------------------------------------------------------

def test_normalization_batch_invariant():
    """Doubling the batch with identical examples must not change the loss."""
    torch.manual_seed(42)
    t_logits = torch.randn(1, 5, 100)
    s_logits = torch.randn(1, 5, 100)
    mask = torch.ones(1, 5, dtype=torch.bool)

    # FKL
    loss_single_fkl = compute_kd_forward_kl(t_logits, s_logits, mask, T=1.0)
    loss_double_fkl = compute_kd_forward_kl(
        t_logits.repeat(2, 1, 1),
        s_logits.repeat(2, 1, 1),
        mask.repeat(2, 1),
        T=1.0,
    )
    assert torch.allclose(loss_single_fkl, loss_double_fkl, atol=1e-5), (
        f"FKL not batch-invariant: single={loss_single_fkl.item()}, "
        f"double={loss_double_fkl.item()}"
    )

    # RKL
    loss_single_rkl = compute_kd_reverse_kl(s_logits, t_logits, mask)
    loss_double_rkl = compute_kd_reverse_kl(
        s_logits.repeat(2, 1, 1),
        t_logits.repeat(2, 1, 1),
        mask.repeat(2, 1),
    )
    assert torch.allclose(loss_single_rkl, loss_double_rkl, atol=1e-5), (
        f"RKL not batch-invariant: single={loss_single_rkl.item()}, "
        f"double={loss_double_rkl.item()}"
    )


# ------------------------------------------------------------------
# Test 4: RKL invariant to temperature (via compute_total_loss)
# ------------------------------------------------------------------

def test_rkl_invariant_to_temperature():
    """RKL does not use temperature — total loss must be identical for T=1 and T=4."""
    teacher_logits, student_logits, labels, attention_mask = _make_toy_data()

    loss_T1 = compute_total_loss(
        teacher_logits, student_logits, labels, attention_mask,
        T=1.0, alpha=0.5, kd_mode='rkl',
    )[0]
    loss_T4 = compute_total_loss(
        teacher_logits, student_logits, labels, attention_mask,
        T=4.0, alpha=0.5, kd_mode='rkl',
    )[0]

    assert torch.allclose(loss_T1, loss_T4, atol=1e-5), (
        f"RKL should be T-invariant: T=1 -> {loss_T1.item()}, T=4 -> {loss_T4.item()}"
    )


# ------------------------------------------------------------------
# Test 5: RKL non-negative
# ------------------------------------------------------------------

def test_rkl_non_negative():
    """Reverse KL divergence must be >= 0."""
    torch.manual_seed(42)
    for seed in [0, 1, 2, 3, 42]:
        g = torch.Generator().manual_seed(seed)
        s_logits = torch.randn(2, 5, 100, generator=g)
        t_logits = torch.randn(2, 5, 100, generator=g)
        mask = torch.ones(2, 5, dtype=torch.bool)

        loss = compute_kd_reverse_kl(s_logits, t_logits, mask)
        assert loss.item() >= -1e-6, (
            f"RKL should be >= 0, got {loss.item()} (seed={seed})"
        )


# ------------------------------------------------------------------
# Test 6: FKL sensitive to temperature
# ------------------------------------------------------------------

def test_fkl_sensitive_to_temperature():
    """FKL total loss must differ when temperature changes."""
    teacher_logits, student_logits, labels, attention_mask = _make_toy_data()

    loss_T1 = compute_total_loss(
        teacher_logits, student_logits, labels, attention_mask,
        T=1.0, alpha=0.5, kd_mode='fkl',
    )[0]
    loss_T4 = compute_total_loss(
        teacher_logits, student_logits, labels, attention_mask,
        T=4.0, alpha=0.5, kd_mode='fkl',
    )[0]

    assert not torch.allclose(loss_T1, loss_T4, atol=1e-3), (
        f"FKL should vary with T: T=1 -> {loss_T1.item()}, T=4 -> {loss_T4.item()}"
    )
