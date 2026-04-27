"""
Knowledge Distillation losses for Causal LM (token-level).

References
----------
Hinton, Vinyals & Dean (2015) – "Distilling the Knowledge in a Neural Network"
Gu et al. (2024) – "MiniLLM: Knowledge Distillation of Large Language Models"

Notation
--------
* z^T, z^S : teacher / student logits, shape [B, L, V]
* y        : labels, shape [B, L], with -100 for positions to ignore
* T        : temperature (T > 0)
* α        : mixing coefficient (fixed at 0.5)

Forward KL (Hinton 2015):
    p^T  = softmax(z^T / T),  p^S  = softmax(z^S / T)
    L_FKL = (1/|M|) Σ KL(p^T || p^S)
    Combined: L = α·L_CE + (1−α)·T²·L_FKL

Reverse KL (MiniLLM, Gu et al. 2024):
    p_S = softmax(z^S),  p_T = softmax(z^T)   ← T=1 (no temperature)
    L_RKL = (1/|M|) Σ KL(p_S || p_T)
    Combined: L = α·L_CE + (1−α)·L_RKL        ← no T² factor

CE Baseline:
    L = L_CE

Mask M:
    After causal shift (next-token prediction), positions where
    labels ≠ -100  AND  attention_mask == 1.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# 5.1  Causal shift
# ------------------------------------------------------------------

def shift_for_causal_lm(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the standard next-token shift for causal language modelling.

    Returns
    -------
    shift_logits : [B, L-1, V]
    shift_labels : [B, L-1]
    valid_mask   : [B, L-1]  bool – True where label is valid & attended
    """
    assert logits.dim() == 3, f"logits must be [B, L, V], got {logits.shape}"
    assert labels.dim() == 2, f"labels must be [B, L], got {labels.shape}"
    assert attention_mask.dim() == 2, f"attention_mask must be [B, L], got {attention_mask.shape}"
    assert logits.size(0) == labels.size(0) == attention_mask.size(0)
    assert logits.size(1) == labels.size(1) == attention_mask.size(1)

    shift_logits = logits[:, :-1, :].contiguous()       # [B, L-1, V]
    shift_labels = labels[:, 1:].contiguous()            # [B, L-1]
    shift_mask = attention_mask[:, 1:].contiguous()      # [B, L-1]

    valid_mask = shift_mask.bool() & (shift_labels != -100)  # [B, L-1]

    return shift_logits, shift_labels, valid_mask


# ------------------------------------------------------------------
# 5.2  Hard-label cross-entropy
# ------------------------------------------------------------------

def compute_ce(
    shift_student_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Token-level CE with masking, normalised by number of valid tokens.

    Returns scalar tensor (0.0 when no valid tokens).
    """
    B, L_minus1, V = shift_student_logits.shape
    assert shift_labels.shape == (B, L_minus1)
    assert valid_mask.shape == (B, L_minus1)

    n_valid = valid_mask.sum()
    if n_valid == 0:
        return shift_student_logits.new_tensor(0.0)

    # Flatten for F.cross_entropy
    flat_logits = shift_student_logits.view(-1, V)           # [B*(L-1), V]
    flat_labels = shift_labels.view(-1)                       # [B*(L-1)]
    flat_mask = valid_mask.view(-1)                           # [B*(L-1)]

    ce_all = F.cross_entropy(flat_logits, flat_labels, reduction="none")  # [B*(L-1)]
    ce_masked = (ce_all * flat_mask.float()).sum()
    loss_ce = ce_masked / n_valid.float()

    return loss_ce


# ------------------------------------------------------------------
# 5.3  Forward KL divergence (KD)
# ------------------------------------------------------------------

def compute_kd_forward_kl(
    shift_teacher_logits: torch.Tensor,
    shift_student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    T: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Forward KL(p_teacher || p_student) averaged over valid tokens.

    Returns scalar tensor (0.0 when no valid tokens).
    """
    assert shift_teacher_logits.shape == shift_student_logits.shape
    B, L_minus1, V = shift_teacher_logits.shape
    assert valid_mask.shape == (B, L_minus1)

    n_valid = valid_mask.sum()
    if n_valid == 0:
        return shift_teacher_logits.new_tensor(0.0)

    # Compute in float32 for numerical stability
    t_logits = shift_teacher_logits.float() / T
    s_logits = shift_student_logits.float() / T

    p_teacher = F.softmax(t_logits, dim=-1)                    # [B, L-1, V]
    logp_student = F.log_softmax(s_logits, dim=-1)             # [B, L-1, V]
    logp_teacher = torch.log(p_teacher.clamp_min(eps))         # [B, L-1, V]

    # KL per token: sum over vocab
    kl_per_token = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1)  # [B, L-1]

    kl_masked = (kl_per_token * valid_mask.float()).sum()
    loss_kd = kl_masked / n_valid.float()

    return loss_kd


# ------------------------------------------------------------------
# 5.3b  Reverse KL divergence (RKL) – MiniLLM (Gu et al. 2024)
# ------------------------------------------------------------------

def compute_kd_reverse_kl(
    shift_student_logits: torch.Tensor,   # [B, L-1, V]
    shift_teacher_logits: torch.Tensor,   # [B, L-1, V]
    valid_mask: torch.Tensor,             # [B, L-1] bool
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reverse KL(p_S || p_T) averaged over valid tokens.

    Source: MiniLLM (Gu et al. 2024).
    NO temperature. NO T² factor. Normalised by number of valid tokens.

    NOTE: argument order is (student, teacher) – student FIRST,
    matching KL(p_S || p_T) where p_S is the "source" distribution.
    This is REVERSED compared to compute_kd_forward_kl(teacher, student).
    """
    assert shift_student_logits.shape == shift_teacher_logits.shape
    B, L_minus1, V = shift_student_logits.shape
    assert valid_mask.shape == (B, L_minus1)

    n_valid = valid_mask.sum()
    if n_valid == 0:
        return shift_student_logits.new_tensor(0.0)

    # T=1: no temperature division
    s_logits = shift_student_logits.float()
    t_logits = shift_teacher_logits.float()

    p_student = F.softmax(s_logits, dim=-1)              # [B, L-1, V]
    logp_student = F.log_softmax(s_logits, dim=-1)       # [B, L-1, V]
    logp_teacher = torch.log(
        F.softmax(t_logits, dim=-1).clamp_min(eps)       # [B, L-1, V]
    )

    # KL(p_S || p_T) = Σ_v p_S(v) * [log p_S(v) - log p_T(v)]
    kl_per_token = (p_student * (logp_student - logp_teacher)).sum(dim=-1)  # [B, L-1]

    kl_masked = (kl_per_token * valid_mask.float()).sum()
    return kl_masked / n_valid.float()


# ------------------------------------------------------------------
# 5.4  Combined loss
# ------------------------------------------------------------------

def compute_total_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    T: float,
    alpha: float,
    kd_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute the full KD + CE loss with explicit causal shift.

    Returns
    -------
    loss_total     : scalar
    loss_ce        : scalar
    loss_kd        : scalar (0.0 for ce_only mode)
    n_valid_tokens : int

    Modes
    -----
    ce_only : L = L_CE
    fkl     : L = α·L_CE + (1−α)·T²·L_FKL
    rkl     : L = α·L_CE + (1−α)·L_RKL   (no T²)
    """
    shift_s, shift_labels, valid_mask = shift_for_causal_lm(
        student_logits, labels, attention_mask,
    )

    loss_ce = compute_ce(shift_s, shift_labels, valid_mask)

    if kd_mode == 'ce_only':
        loss_kd = shift_s.new_tensor(0.0)
        loss_total = loss_ce

    elif kd_mode == 'fkl':
        shift_t, _, _ = shift_for_causal_lm(
            teacher_logits, labels, attention_mask,
        )
        loss_kd = compute_kd_forward_kl(shift_t, shift_s, valid_mask, T)
        loss_total = alpha * loss_ce + (1 - alpha) * (T ** 2) * loss_kd

    elif kd_mode == 'rkl':
        shift_t, _, _ = shift_for_causal_lm(
            teacher_logits, labels, attention_mask,
        )
        # NOTE: argument order is (student, teacher) for reverse KL
        loss_kd = compute_kd_reverse_kl(shift_s, shift_t, valid_mask)
        loss_total = alpha * loss_ce + (1 - alpha) * loss_kd  # NO T²

    else:
        raise ValueError(f'kd_mode must be ce_only | fkl | rkl, got {kd_mode!r}')

    n_valid = int(valid_mask.sum().item())
    return loss_total, loss_ce, loss_kd, n_valid
