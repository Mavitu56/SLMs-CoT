"""
Per-region weighted variants of the KD losses.

This module does NOT modify ``losses_kd.py`` (which is frozen). It reuses the
original ``shift_for_causal_lm`` and reimplements CE / FKL / RKL with an
explicit per-token weight derived from ``region_ids``. With uniform weights it
reduces *exactly* to ``compute_total_loss`` in ``losses_kd.py``.

Motivation
----------
GSM8K-CoT has a ~17.5:1 reasoning:answer token ratio, so 94.6% of the
loss-bearing positions are reasoning tokens. Reviewers asked whether this
imbalance is the causal mechanism behind CE > KD. Up-weighting answer tokens
(region 2) by ~17.5x rebalances the per-token gradient and tests that
hypothesis directly.

Math
----
Each loss is a *weighted* mean over valid tokens:

    L = sum_t ( w_t * l_t ) / sum_t ( w_t )

where l_t is the per-token loss (CE, forward-KL, or reverse-KL), and
w_t = region_weight[region_id_t] for valid tokens (labels != -100 and attended),
0 otherwise. When every region weight is 1.0, sum_t(w_t) = |M| and the
expression collapses to the original (1/|M|) * sum_t l_t.

Region ids (from data_gsm8k): 0 = prompt (already masked), 1 = reasoning,
2 = answer. Regions absent from ``region_weights`` default to weight 1.0.

References
----------
Hinton, Vinyals & Dean (2015) - Forward KL with T^2 scaling.
Gu et al. (2024), MiniLLM - Reverse KL, no temperature, no T^2.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# Reuse the frozen causal-shift helper - losses_kd.py is NOT modified.
from src.losses.losses_kd import shift_for_causal_lm


def _per_token_weight(
    shift_region_ids: torch.Tensor,   # [B, L-1] long (-1 = pad)
    valid_mask: torch.Tensor,         # [B, L-1] bool
    region_weights: Dict[int, float],
) -> torch.Tensor:
    """Build a [B, L-1] float weight tensor: region_weight on valid tokens, 0 else.

    Regions not present in ``region_weights`` default to 1.0.
    """
    w = torch.ones_like(shift_region_ids, dtype=torch.float32)
    for rid, wt in region_weights.items():
        w = torch.where(
            shift_region_ids == int(rid),
            torch.as_tensor(float(wt), dtype=torch.float32, device=w.device),
            w,
        )
    return w * valid_mask.float()


def _weighted_ce(
    shift_student_logits: torch.Tensor,   # [B, L-1, V]
    shift_labels: torch.Tensor,           # [B, L-1]
    w: torch.Tensor,                      # [B, L-1] float (0 where invalid)
    eps: float = 1e-8,
) -> torch.Tensor:
    B, Lm1, V = shift_student_logits.shape
    denom = w.sum()
    if denom <= 0:
        return shift_student_logits.new_tensor(0.0)
    ce_all = F.cross_entropy(
        shift_student_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
    ).view(B, Lm1)
    # Labels at invalid positions may be -100; w is 0 there, so they contribute 0.
    ce_all = torch.nan_to_num(ce_all, nan=0.0, posinf=0.0, neginf=0.0)
    return (ce_all * w).sum() / denom.clamp_min(eps)


def _weighted_fkl(
    shift_teacher_logits: torch.Tensor,
    shift_student_logits: torch.Tensor,
    w: torch.Tensor,
    T: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    denom = w.sum()
    if denom <= 0:
        return shift_teacher_logits.new_tensor(0.0)
    t = shift_teacher_logits.float() / T
    s = shift_student_logits.float() / T
    p_t = F.softmax(t, dim=-1)
    logp_s = F.log_softmax(s, dim=-1)
    logp_t = torch.log(p_t.clamp_min(eps))
    kl = (p_t * (logp_t - logp_s)).sum(dim=-1)           # [B, L-1]
    return (kl * w).sum() / denom.clamp_min(eps)


def _weighted_rkl(
    shift_student_logits: torch.Tensor,
    shift_teacher_logits: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    denom = w.sum()
    if denom <= 0:
        return shift_student_logits.new_tensor(0.0)
    s = shift_student_logits.float()
    t = shift_teacher_logits.float()
    p_s = F.softmax(s, dim=-1)
    logp_s = F.log_softmax(s, dim=-1)
    logp_t = torch.log(F.softmax(t, dim=-1).clamp_min(eps))
    kl = (p_s * (logp_s - logp_t)).sum(dim=-1)           # [B, L-1]
    return (kl * w).sum() / denom.clamp_min(eps)


def compute_total_loss_weighted(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    region_ids: torch.Tensor,            # [B, L] long, BEFORE causal shift
    T: float,
    alpha: float,
    kd_mode: str,
    region_weights: Dict[int, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Region-weighted version of ``compute_total_loss``.

    The same per-token weight ``w`` is applied to BOTH the CE and KD terms, so
    a single knob (``region_weights``) controls the reasoning/answer balance of
    the whole objective. With ``region_weights`` all 1.0 this returns values
    numerically identical to the frozen ``compute_total_loss``.

    Returns
    -------
    loss_total, loss_ce, loss_kd, n_valid_tokens
    """
    shift_s, shift_labels, valid_mask = shift_for_causal_lm(
        student_logits, labels, attention_mask,
    )
    # Align region_ids with the next-token shift (drop first position).
    shift_region = region_ids[:, 1:].contiguous()
    w = _per_token_weight(shift_region, valid_mask, region_weights)

    loss_ce = _weighted_ce(shift_s, shift_labels, w)

    if kd_mode == "ce_only":
        loss_kd = shift_s.new_tensor(0.0)
        loss_total = loss_ce

    elif kd_mode == "fkl":
        shift_t, _, _ = shift_for_causal_lm(teacher_logits, labels, attention_mask)
        loss_kd = _weighted_fkl(shift_t, shift_s, w, T)
        loss_total = alpha * loss_ce + (1 - alpha) * (T ** 2) * loss_kd

    elif kd_mode == "rkl":
        shift_t, _, _ = shift_for_causal_lm(teacher_logits, labels, attention_mask)
        loss_kd = _weighted_rkl(shift_s, shift_t, w)   # (student, teacher) order
        loss_total = alpha * loss_ce + (1 - alpha) * loss_kd

    else:
        raise ValueError(f"kd_mode must be ce_only | fkl | rkl, got {kd_mode!r}")

    n_valid = int(valid_mask.sum().item())
    return loss_total, loss_ce, loss_kd, n_valid
