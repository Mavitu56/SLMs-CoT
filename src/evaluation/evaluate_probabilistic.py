"""
Probabilistic evaluation of a student model (with optional teacher).

Runs a full eval-mode pass over a DataLoader, collecting:
  • mean entropy, NLL, perplexity
  • mean max-probability (sharpness)
  • mean KL(teacher ‖ student)  (only when teacher is provided)
  • ECE  ← computed over the full dataset in a single pass (not per-batch average)
  • per-position curves: entropy, maxprob, KL
  • **all scalar metrics segmented by region**:
        prompt (region 0), reasoning (region 1), answer (region 2)

Region segmentation avoids diluting reasoning/answer metrics with
the highly-predictable prompt tokens.

All computation is in **eval mode** with ``torch.no_grad()``.
No Trainer, no external calibration libraries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.losses.losses_kd import shift_for_causal_lm
from src.evaluation.analysis_metrics import (
    mean_entropy,
    mean_max_probability,
    mean_kl,
    mean_nll,
    token_entropy,
    token_max_probability,
)


# ------------------------------------------------------------------
# Region constants (dataset-agnostic, local definitions)
# ------------------------------------------------------------------
REGION_PROMPT    = 0
REGION_REASONING = 1
REGION_ANSWER    = 2

# Human-readable region names (for JSON keys)
_REGION_NAMES = {
    REGION_PROMPT: "prompt",
    REGION_REASONING: "reasoning",
    REGION_ANSWER: "answer",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _align_vocab(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate teacher vocab dim to match student if needed."""
    if teacher_logits.size(-1) > student_logits.size(-1):
        teacher_logits = teacher_logits[..., : student_logits.size(-1)]
    assert teacher_logits.shape[-1] == student_logits.shape[-1], (
        f"Vocab mismatch: teacher {teacher_logits.shape[-1]} "
        f"!= student {student_logits.shape[-1]}"
    )
    return teacher_logits, student_logits


def verify_same_tokenizer(
    teacher_name: str,
    student_name: str,
    n_check: int = 1000,
) -> None:
    """Assert that teacher and student share the same token mapping.

    Loads both tokenizers, checks that the first ``n_check`` token ids
    map to identical strings, and raises ``AssertionError`` on mismatch.
    """
    tok_t = AutoTokenizer.from_pretrained(teacher_name)
    tok_s = AutoTokenizer.from_pretrained(student_name)
    V = min(tok_t.vocab_size, tok_s.vocab_size, n_check)
    for i in range(V):
        t_tok = tok_t.convert_ids_to_tokens(i)
        s_tok = tok_s.convert_ids_to_tokens(i)
        assert t_tok == s_tok, (
            f"Tokenizer mismatch at id {i}: teacher='{t_tok}' vs student='{s_tok}'. "
            f"Teacher and student MUST share the same tokenizer for KL metrics."
        )
    print(f"[tokenizer] verified {V} token ids match between teacher and student ✓")


def _build_region_mask(
    shifted_region_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    region_id: int,
) -> torch.Tensor:
    """Return a bool mask that is True only where region matches AND valid."""
    return valid_mask & (shifted_region_ids == region_id)


def _ece_from_flat(
    flat_conf: torch.Tensor,
    flat_correct: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """ECE computed once from pre-accumulated flat confidence and correctness tensors.

    This is the correct approach: accumulate (confidence, correct) pairs across
    all batches, then call this function once so that all tokens are binned
    together. Computing ECE per-batch and averaging is NOT equivalent because
    within-bin statistics (acc, conf) depend on the full bin population.

    Parameters
    ----------
    flat_conf    : [N] float – top-1 confidence for each valid token
    flat_correct : [N] float – 1.0 if top-1 prediction was correct, else 0.0
    n_bins       : equal-width bins in [0, 1] (default 10, per spec A1)

    Returns
    -------
    Scalar tensor in [0, 1].
    """
    N = flat_conf.numel()
    if N == 0:
        return torch.tensor(0.0)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.tensor(0.0)

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = (
            (flat_conf >= lo) & (flat_conf <= hi)  # last bin: inclusive on both sides
            if i == n_bins - 1
            else (flat_conf >= lo) & (flat_conf < hi)
        )
        n_in_bin = in_bin.sum().item()
        if n_in_bin == 0:
            continue
        avg_conf = flat_conf[in_bin].mean()
        avg_acc  = flat_correct[in_bin].mean()
        ece = ece + (n_in_bin / N) * (avg_conf - avg_acc).abs()

    return ece.clamp(0.0, 1.0)


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    student: AutoModelForCausalLM,
    teacher: Optional[AutoModelForCausalLM],
    dataloader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run probabilistic evaluation over the full dataloader.

    Parameters
    ----------
    student    : The student model (will be set to eval mode).
    teacher    : The teacher model, or ``None`` when no KL is needed.
    dataloader : DataLoader (already tokenised & collated).
                 Should include ``region_ids`` in each batch for
                 per-region breakdown (gracefully skipped if absent).
    cfg        : Config dict – must contain ``max_length``.

    Returns
    -------
    dict with keys:
        mean_entropy, mean_maxprob, mean_nll, ppl,
        mean_kl (or None), ece,
        region_prompt, region_reasoning, region_answer
            (each: {entropy, maxprob, nll, ppl, kl, ece, n_tokens}),
        kl_per_position, ent_per_position, mp_per_position,
        n_tokens_eval, n_batches, avg_seq_length
    """
    student.eval()
    if teacher is not None:
        teacher.eval()

    device = next(student.parameters()).device

    # --- Global accumulators (weighted by n_valid per batch) ---
    total_entropy = 0.0
    total_maxprob = 0.0
    total_nll     = 0.0
    total_kl      = 0.0
    total_tokens  = 0
    total_sequences  = 0    # actual number of sequences (for avg_seq_length)
    total_seq_length = 0    # sum of non-pad token counts
    n_batches = 0

    # ECE: accumulate (confidence, correct) across all batches, compute once at end
    all_conf:    List[torch.Tensor] = []
    all_correct: List[torch.Tensor] = []

    # --- Per-region accumulators ---
    region_acc: Dict[int, Dict[str, float]] = {}
    region_conf:    Dict[int, List[torch.Tensor]] = {}
    region_correct: Dict[int, List[torch.Tensor]] = {}
    for rid in (REGION_PROMPT, REGION_REASONING, REGION_ANSWER):
        region_acc[rid] = {
            "entropy": 0.0, "maxprob": 0.0, "nll": 0.0,
            "kl": 0.0, "n_tokens": 0,
        }
        region_conf[rid]    = []
        region_correct[rid] = []

    # --- Per-position accumulators (response-relative indexing) ---
    max_resp_pos = cfg["max_length"] - 1   # max possible response length
    kl_pos_sum  = torch.zeros(max_resp_pos, dtype=torch.float32, device=device)
    kl_pos_cnt  = torch.zeros(max_resp_pos, dtype=torch.float32, device=device)
    ent_pos_sum = torch.zeros(max_resp_pos, dtype=torch.float32, device=device)
    ent_pos_cnt = torch.zeros(max_resp_pos, dtype=torch.float32, device=device)
    mp_pos_sum  = torch.zeros(max_resp_pos, dtype=torch.float32, device=device)
    mp_pos_cnt  = torch.zeros(max_resp_pos, dtype=torch.float32, device=device)

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # ---- Student forward ----
        s_out = student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        s_logits = s_out.logits.detach()

        # ---- Teacher forward (if available) ----
        t_logits = None
        if teacher is not None:
            t_out = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            t_logits = t_out.logits.detach()
            t_logits, s_logits = _align_vocab(t_logits, s_logits)

        # ---- Causal shift ----
        shift_s, shift_labels, valid_mask = shift_for_causal_lm(
            s_logits, batch["labels"], batch["attention_mask"],
        )
        shift_t = None
        if t_logits is not None:
            shift_t, _, _ = shift_for_causal_lm(
                t_logits, batch["labels"], batch["attention_mask"],
            )

        # Shift region_ids the same way (drop first, keep [1:])
        has_regions = "region_ids" in batch
        shifted_region_ids = None
        if has_regions:
            shifted_region_ids = batch["region_ids"][:, 1:].contiguous()

        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            continue

        # ---- Sequence count & average length ----
        total_sequences  += batch["input_ids"].shape[0]
        total_seq_length += int(batch["attention_mask"].sum().item())

        # ---- Compute probs/preds once — reused for global and per-region ECE ----
        _probs           = F.softmax(shift_s.float(), dim=-1)    # [B, L-1, V]
        _max_prob, _preds = _probs.max(dim=-1)                    # [B, L-1]

        # ---- Global scalar metrics ----
        batch_entropy = mean_entropy(shift_s, valid_mask).item()
        batch_maxprob = mean_max_probability(shift_s, valid_mask).item()
        batch_nll     = mean_nll(shift_s, shift_labels, valid_mask).item()

        total_entropy += batch_entropy * n_valid
        total_maxprob += batch_maxprob * n_valid
        total_nll     += batch_nll     * n_valid
        total_tokens  += n_valid

        # Accumulate confidence and correctness for global ECE
        all_conf.append(_max_prob[valid_mask].cpu())
        all_correct.append((_preds[valid_mask] == shift_labels[valid_mask]).float().cpu())

        if shift_t is not None:
            batch_kl = mean_kl(shift_t, shift_s, valid_mask).item()
            total_kl += batch_kl * n_valid

        # ---- Per-region scalar metrics ----
        if has_regions and shifted_region_ids is not None:
            for rid in (REGION_PROMPT, REGION_REASONING, REGION_ANSWER):
                rmask = _build_region_mask(shifted_region_ids, valid_mask, rid)
                rn = int(rmask.sum().item())
                if rn == 0:
                    continue
                racc = region_acc[rid]
                racc["entropy"] += mean_entropy(shift_s, rmask).item() * rn
                racc["maxprob"] += mean_max_probability(shift_s, rmask).item() * rn
                racc["nll"]     += mean_nll(shift_s, shift_labels, rmask).item() * rn
                racc["n_tokens"] += rn
                if shift_t is not None:
                    racc["kl"] += mean_kl(shift_t, shift_s, rmask).item() * rn
                # Accumulate per-region ECE data
                region_conf[rid].append(_max_prob[rmask].cpu())
                region_correct[rid].append((_preds[rmask] == shift_labels[rmask]).float().cpu())

        # ---- Per-position curves (response-relative indexing) ----
        B_cur = shift_s.size(0)

        # Compute per-token values once for the whole batch (GPU)
        ent_tokens = token_entropy(shift_s)            # [B, L_cur]
        mp_tokens  = token_max_probability(shift_s)    # [B, L_cur]
        kl_tokens_batch = None
        if shift_t is not None:
            p_t = F.softmax(shift_t.float(), dim=-1)
            logp_t = torch.log(p_t.clamp_min(1e-8))
            logp_s = F.log_softmax(shift_s.float(), dim=-1)
            kl_tokens_batch = (p_t * (logp_t - logp_s)).sum(dim=-1).clamp_min(0.0)

        for b in range(B_cur):
            sample_mask = valid_mask[b]
            valid_positions = sample_mask.nonzero(as_tuple=False).squeeze(-1)
            if valid_positions.numel() == 0:
                continue
            resp_start = valid_positions[0].item()

            for abs_pos in valid_positions.tolist():
                rel_pos = abs_pos - resp_start
                if rel_pos >= max_resp_pos:
                    break
                ent_pos_sum[rel_pos] += ent_tokens[b, abs_pos].item()
                ent_pos_cnt[rel_pos] += 1.0
                mp_pos_sum[rel_pos]  += mp_tokens[b, abs_pos].item()
                mp_pos_cnt[rel_pos]  += 1.0
                if kl_tokens_batch is not None:
                    kl_pos_sum[rel_pos] += kl_tokens_batch[b, abs_pos].item()
                    kl_pos_cnt[rel_pos] += 1.0

        n_batches += 1

    # ---- Aggregate ----
    if total_tokens == 0:
        raise RuntimeError("No valid tokens found in the evaluation set.")

    agg_entropy = total_entropy / total_tokens
    agg_maxprob = total_maxprob / total_tokens
    agg_nll     = total_nll     / total_tokens
    agg_ppl     = math.exp(agg_nll)
    agg_kl      = (total_kl / total_tokens) if teacher is not None else None

    # Global ECE: computed once over all accumulated tokens (correct approach)
    flat_conf    = torch.cat(all_conf)    if all_conf    else torch.tensor([])
    flat_correct = torch.cat(all_correct) if all_correct else torch.tensor([])
    agg_ece = _ece_from_flat(flat_conf, flat_correct).item()

    avg_seq_len = total_seq_length / max(total_sequences, 1)

    # Per-position curves → list
    def _curve_to_list(s: torch.Tensor, c: torch.Tensor):
        safe_c = c.clamp_min(1.0)
        curve = (s / safe_c).cpu().tolist()
        nonzero = (c > 0).nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            return []
        last_idx = nonzero[-1].item()
        return curve[: last_idx + 1]

    # Per-region aggregation
    def _aggregate_region(rid: int) -> Dict[str, float | None]:
        racc = region_acc[rid]
        rn = racc["n_tokens"]
        if rn == 0:
            return {
                "entropy": None, "maxprob": None,
                "nll": None, "ppl": None,
                "kl": None, "ece": None, "n_tokens": 0,
            }
        r_nll = racc["nll"] / rn
        # Per-region ECE: computed once from all accumulated tokens in this region
        r_flat_conf    = torch.cat(region_conf[rid])    if region_conf[rid]    else torch.tensor([])
        r_flat_correct = torch.cat(region_correct[rid]) if region_correct[rid] else torch.tensor([])
        r_ece = _ece_from_flat(r_flat_conf, r_flat_correct).item()
        return {
            "entropy": round(racc["entropy"] / rn, 6),
            "maxprob": round(racc["maxprob"] / rn, 6),
            "nll":     round(r_nll, 6),
            "ppl":     round(math.exp(r_nll), 4),
            "kl":      round(racc["kl"] / rn, 6) if teacher is not None else None,
            "ece":     round(r_ece, 6),
            "n_tokens": rn,
        }

    results = {
        # Global scalars
        "mean_entropy": round(agg_entropy, 6),
        "mean_maxprob": round(agg_maxprob, 6),
        "mean_nll":     round(agg_nll, 6),
        "ppl":          round(agg_ppl, 4),
        "mean_kl":      round(agg_kl, 6) if agg_kl is not None else None,
        "ece":          round(agg_ece, 6),
        # Per-region
        "region_prompt":    _aggregate_region(REGION_PROMPT),
        "region_reasoning": _aggregate_region(REGION_REASONING),
        "region_answer":    _aggregate_region(REGION_ANSWER),
        # Per-position curves
        "kl_per_position": (
            _curve_to_list(kl_pos_sum, kl_pos_cnt)
            if teacher is not None else None
        ),
        "ent_per_position": _curve_to_list(ent_pos_sum, ent_pos_cnt),
        "mp_per_position":  _curve_to_list(mp_pos_sum, mp_pos_cnt),
        # Meta
        "n_tokens_eval": total_tokens,
        "n_batches":     n_batches,
        "avg_seq_length": round(avg_seq_len, 1),
    }

    # ---- Print summary ----
    print(f"\n{'='*62}")
    print(f" Probabilistic Evaluation  ({n_batches} batches, "
          f"{total_tokens} tokens, avg_len={avg_seq_len:.1f})")
    print(f"{'='*62}")
    print(f"  Mean Entropy     : {results['mean_entropy']:.6f}")
    print(f"  Mean MaxProb     : {results['mean_maxprob']:.6f}")
    print(f"  Mean NLL         : {results['mean_nll']:.6f}")
    print(f"  Perplexity       : {results['ppl']:.4f}")
    if results["mean_kl"] is not None:
        print(f"  Mean KL(T||S)    : {results['mean_kl']:.6f}")
    print(f"  ECE              : {results['ece']:.6f}")

    # Region breakdown
    for rname, rkey in [("Prompt",    "region_prompt"),
                        ("Reasoning", "region_reasoning"),
                        ("Answer",    "region_answer")]:
        rd = results[rkey]
        n_tok = rd["n_tokens"]
        if n_tok == 0:
            print(f"  [{rname:>9}]      : (no tokens)")
            continue
        kl_str = f"  kl={rd['kl']:.4f}" if rd["kl"] is not None else ""
        print(
            f"  [{rname:>9}] ({n_tok:>6} tok)  "
            f"H={rd['entropy']:.4f}  mp={rd['maxprob']:.4f}  "
            f"nll={rd['nll']:.4f}  ppl={rd['ppl']:.2f}  "
            f"ece={rd['ece']:.4f}{kl_str}"
        )
    print(f"{'='*62}\n")

    return results
