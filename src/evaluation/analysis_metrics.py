"""
Probabilistic analysis metrics for Causal LM evaluation (Layer 2).

This module implements five families of metrics to compare the
probabilistic behaviour of student models trained under different
objectives (CE-only, KD-only, KD+CE) on GSM8K.

Mathematical Definitions
========================

1. **Token-level Entropy**

       H(p)_t = −Σ_v  p_t(v) · log p_t(v)

   where p_t = softmax(z_t) and the sum runs over the full vocabulary V.
   A *sharp* distribution (low entropy) concentrates mass on few tokens;
   a *smooth* distribution (high entropy) spreads mass more evenly.

2. **Sharpness (Maximum Probability)**

       maxprob_t = max_v  p_t(v)

   Complementary view of sharpness: higher maxprob ↔ lower entropy.

3. **Forward KL Divergence (Teacher → Student) per Position**

       KL(P^T_t ‖ P^S_t) = Σ_v  p^T_t(v) · [log p^T_t(v) − log p^S_t(v)]

   Computed *after* the standard causal-LM shift and averaged over the
   batch dimension while preserving the sequence-position dimension,
   so the result is a 1-D curve  position → KL.

4. **Expected Calibration Error (ECE)**

       ECE = Σ_b  (|B_b| / N) · |acc(B_b) − conf(B_b)|

   where tokens are binned by their confidence (maxprob), and
   acc / conf are the mean accuracy / confidence within each bin.

5. **Negative Log-Likelihood & Perplexity**

       NLL_t = −log p_t(y_t)
       Mean NLL = (1/|M|) Σ_{t∈M} NLL_t
       PPL = exp(Mean NLL)

   Standard language-modelling metrics.  More stable and traditional
   than ECE for evaluating token-level calibration in LMs.

Implementation Notes
--------------------
* All soft distributions are computed in **float32** regardless of
  the model's native dtype, for numerical stability.
* Softmax / log-softmax are computed **once** and reused wherever
  possible to avoid redundant work.
* Every function expects logits that have already been **shifted**
  for causal LM (i.e. logits[:, :-1] predicting labels[:, 1:]),
  and a ``valid_mask`` boolean tensor of the same [B, L] shape
  (True where the token is valid for evaluation).
* Vocab-size alignment between teacher and student must be
  handled *before* calling these functions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ==================================================================
# 1.  Token-level Entropy
# ==================================================================

def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-token entropy of the predicted distribution.

    Parameters
    ----------
    logits : Tensor [B, L, V]
        Raw (shifted) logits from the model.

    Returns
    -------
    Tensor [B, L]
        H(p)_t for every token position.
    """
    assert logits.dim() == 3, f"logits must be [B, L, V], got {logits.shape}"

    # Cast to float32 for numerical stability
    logits_f32 = logits.float()
    log_probs = F.log_softmax(logits_f32, dim=-1)     # [B, L, V]
    probs = log_probs.exp()                            # reuse log_softmax → exp is cheap

    entropy = -(probs * log_probs).sum(dim=-1)         # [B, L]

    # Sanity: entropy must be non-negative (up to tiny float noise)
    assert (entropy >= -1e-6).all(), (
        f"Negative entropy detected: min={entropy.min().item():.6e}"
    )
    return entropy.clamp_min(0.0)


def mean_entropy(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Scalar mean entropy over all valid tokens.

    Parameters
    ----------
    logits     : Tensor [B, L, V]
    valid_mask : Tensor [B, L]  (bool)

    Returns
    -------
    Scalar tensor (float32).
    """
    assert logits.shape[:2] == valid_mask.shape, (
        f"Shape mismatch: logits {logits.shape[:2]} vs mask {valid_mask.shape}"
    )
    ent = token_entropy(logits)                        # [B, L]
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return logits.new_tensor(0.0, dtype=torch.float32)
    return (ent * valid_mask.float()).sum() / n_valid.float()


# ==================================================================
# 2.  Sharpness (Maximum Probability)
# ==================================================================

def token_max_probability(logits: torch.Tensor) -> torch.Tensor:
    """Per-token maximum probability.

    Parameters
    ----------
    logits : Tensor [B, L, V]

    Returns
    -------
    Tensor [B, L]
    """
    assert logits.dim() == 3, f"logits must be [B, L, V], got {logits.shape}"
    probs = F.softmax(logits.float(), dim=-1)          # [B, L, V]
    max_prob, _ = probs.max(dim=-1)                    # [B, L]
    return max_prob


def mean_max_probability(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Scalar mean max-probability over valid tokens.

    Parameters
    ----------
    logits     : Tensor [B, L, V]
    valid_mask : Tensor [B, L]  (bool)

    Returns
    -------
    Scalar tensor (float32).
    """
    assert logits.shape[:2] == valid_mask.shape
    mp = token_max_probability(logits)                 # [B, L]
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return logits.new_tensor(0.0, dtype=torch.float32)
    result = (mp * valid_mask.float()).sum() / n_valid.float()

    assert 0.0 <= result.item() <= 1.0, (
        f"mean_max_probability out of range: {result.item():.6e}"
    )
    return result


# ==================================================================
# 3.  KL Divergence per Position (Teacher → Student)
# ==================================================================

def kl_per_position(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Position-wise KL(P^T ‖ P^S) averaged over the batch.

    Parameters
    ----------
    teacher_logits : Tensor [B, L, V]  (already shifted)
    student_logits : Tensor [B, L, V]  (already shifted)
    valid_mask     : Tensor [B, L]     (bool, already shifted)
    eps : float
        Clamping floor for teacher probs before log.

    Returns
    -------
    kl_curve : Tensor [L]
        Mean KL at each position (averaged only over valid tokens in
        the batch at that position; positions with zero valid tokens
        are set to 0).
    """
    assert teacher_logits.shape == student_logits.shape, (
        f"Shape mismatch: teacher {teacher_logits.shape} vs student {student_logits.shape}"
    )
    assert teacher_logits.shape[:2] == valid_mask.shape
    B, L, V = teacher_logits.shape

    # Float32 for stability
    t_f32 = teacher_logits.float()
    s_f32 = student_logits.float()

    p_teacher = F.softmax(t_f32, dim=-1)                       # [B, L, V]
    logp_teacher = torch.log(p_teacher.clamp_min(eps))         # [B, L, V]
    logp_student = F.log_softmax(s_f32, dim=-1)                # [B, L, V]

    kl_tokens = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1)  # [B, L]

    # Sanity: KL should be >= 0 (up to float noise)
    assert (kl_tokens >= -1e-6).all(), (
        f"Negative KL detected: min={kl_tokens.min().item():.6e}"
    )
    kl_tokens = kl_tokens.clamp_min(0.0)

    # Average over batch, position by position (only valid tokens)
    mask_f = valid_mask.float()                                 # [B, L]
    n_valid_per_pos = mask_f.sum(dim=0)                         # [L]
    kl_sum_per_pos = (kl_tokens * mask_f).sum(dim=0)            # [L]

    # Avoid division by zero for positions with no valid tokens
    safe_n = n_valid_per_pos.clamp_min(1.0)
    kl_curve = kl_sum_per_pos / safe_n                          # [L]

    return kl_curve


def mean_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scalar mean KL(P^T ‖ P^S) over all valid tokens.

    Parameters
    ----------
    teacher_logits : Tensor [B, L, V]  (shifted)
    student_logits : Tensor [B, L, V]  (shifted)
    valid_mask     : Tensor [B, L]     (bool, shifted)

    Returns
    -------
    Scalar tensor (float32).
    """
    assert teacher_logits.shape == student_logits.shape
    assert teacher_logits.shape[:2] == valid_mask.shape

    t_f32 = teacher_logits.float()
    s_f32 = student_logits.float()

    p_teacher = F.softmax(t_f32, dim=-1)
    logp_teacher = torch.log(p_teacher.clamp_min(eps))
    logp_student = F.log_softmax(s_f32, dim=-1)

    kl_tokens = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1)  # [B, L]
    kl_tokens = kl_tokens.clamp_min(0.0)

    n_valid = valid_mask.sum()
    if n_valid == 0:
        return teacher_logits.new_tensor(0.0, dtype=torch.float32)
    return (kl_tokens * valid_mask.float()).sum() / n_valid.float()


# ==================================================================
# 4.  Expected Calibration Error (ECE)
# ==================================================================

def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """Expected Calibration Error (ECE) over valid tokens.

    Parameters
    ----------
    logits     : Tensor [B, L, V]  (shifted)
    labels     : Tensor [B, L]     (shifted, ground-truth token ids)
    valid_mask : Tensor [B, L]     (bool)
    n_bins : int
        Number of equal-width confidence bins.

    Returns
    -------
    Scalar tensor (float32) in [0, 1].
    """
    assert logits.shape[:2] == labels.shape == valid_mask.shape, (
        f"Shape mismatch: logits {logits.shape[:2]}, "
        f"labels {labels.shape}, mask {valid_mask.shape}"
    )

    # Flatten to 1-D over valid tokens
    probs = F.softmax(logits.float(), dim=-1)                   # [B, L, V]
    max_probs, preds = probs.max(dim=-1)                        # [B, L]

    flat_conf = max_probs[valid_mask]                            # [N_valid]
    flat_pred = preds[valid_mask]                                # [N_valid]
    flat_label = labels[valid_mask]                              # [N_valid]

    N = flat_conf.numel()
    if N == 0:
        return logits.new_tensor(0.0, dtype=torch.float32)

    correct = (flat_pred == flat_label).float()                  # [N_valid]

    # Bin boundaries: equal-width in [0, 1]
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)

    ece = logits.new_tensor(0.0, dtype=torch.float32)
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]

        # Last bin is inclusive on both sides
        if i == n_bins - 1:
            in_bin = (flat_conf >= lo) & (flat_conf <= hi)
        else:
            in_bin = (flat_conf >= lo) & (flat_conf < hi)

        n_in_bin = in_bin.sum().item()
        if n_in_bin == 0:
            continue

        avg_conf = flat_conf[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        ece = ece + (n_in_bin / N) * (avg_conf - avg_acc).abs()

    # Sanity: ECE must be in [0, 1]
    assert 0.0 <= ece.item() <= 1.0 + 1e-6, (
        f"ECE out of range: {ece.item():.6e}"
    )
    return ece.clamp(0.0, 1.0)


# ==================================================================
# 5.  Entropy per Position (for sequence-level analysis)
# ==================================================================

def entropy_per_position(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Position-wise mean entropy, averaged over the batch.

    Parameters
    ----------
    logits     : Tensor [B, L, V]  (shifted)
    valid_mask : Tensor [B, L]     (bool, shifted)

    Returns
    -------
    Tensor [L]  – mean entropy at each position.
    """
    ent = token_entropy(logits)                                 # [B, L]
    mask_f = valid_mask.float()
    n_per_pos = mask_f.sum(dim=0).clamp_min(1.0)               # [L]
    return (ent * mask_f).sum(dim=0) / n_per_pos                # [L]


# ==================================================================
# 6.  Max-prob per Position (for sequence-level analysis)
# ==================================================================

def maxprob_per_position(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Position-wise mean max-probability, averaged over the batch.

    Parameters
    ----------
    logits     : Tensor [B, L, V]  (shifted)
    valid_mask : Tensor [B, L]     (bool, shifted)

    Returns
    -------
    Tensor [L]  – mean maxprob at each position.
    """
    mp = token_max_probability(logits)                          # [B, L]
    mask_f = valid_mask.float()
    n_per_pos = mask_f.sum(dim=0).clamp_min(1.0)
    return (mp * mask_f).sum(dim=0) / n_per_pos                 # [L]


# ==================================================================
# 7.  Negative Log-Likelihood (NLL) & Perplexity
# ==================================================================

def token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-token negative log-likelihood: −log p(y_t).

    Parameters
    ----------
    logits : Tensor [B, L, V]  (shifted)
    labels : Tensor [B, L]     (shifted, ground-truth token ids)

    Returns
    -------
    Tensor [B, L]  –  NLL for every position.
    """
    assert logits.dim() == 3, f"logits must be [B, L, V], got {logits.shape}"
    assert logits.shape[:2] == labels.shape, (
        f"Shape mismatch: logits {logits.shape[:2]} vs labels {labels.shape}"
    )

    log_probs = F.log_softmax(logits.float(), dim=-1)           # [B, L, V]
    # Gather the log-prob of the correct token
    # labels may contain -100 for ignored positions; replace with 0 for gather
    safe_labels = labels.clamp_min(0).unsqueeze(-1)             # [B, L, 1]
    nll = -log_probs.gather(dim=-1, index=safe_labels).squeeze(-1)  # [B, L]
    return nll


def mean_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Scalar mean NLL over valid tokens.

    Parameters
    ----------
    logits     : Tensor [B, L, V]  (shifted)
    labels     : Tensor [B, L]     (shifted)
    valid_mask : Tensor [B, L]     (bool)

    Returns
    -------
    Scalar tensor (float32).
    """
    assert logits.shape[:2] == labels.shape == valid_mask.shape
    nll = token_nll(logits, labels)
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return logits.new_tensor(0.0, dtype=torch.float32)
    return (nll * valid_mask.float()).sum() / n_valid.float()


def perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Perplexity = exp(mean NLL) over valid tokens.

    Parameters
    ----------
    logits     : Tensor [B, L, V]  (shifted)
    labels     : Tensor [B, L]     (shifted)
    valid_mask : Tensor [B, L]     (bool)

    Returns
    -------
    Scalar tensor (float32).
    """
    return torch.exp(mean_nll(logits, labels, valid_mask))
