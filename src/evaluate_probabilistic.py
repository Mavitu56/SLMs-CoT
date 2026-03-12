"""
Probabilistic evaluation of a student model (with optional teacher).

Runs a full eval-mode pass over a DataLoader, collecting:
  • mean entropy, NLL, perplexity
  • mean max-probability (sharpness)
  • mean KL(teacher ‖ student)  (only when teacher is provided)
  • ECE
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
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.losses_kd import shift_for_causal_lm
from src.analysis_metrics import (
    mean_entropy,
    mean_max_probability,
    mean_kl,
    mean_nll,
    perplexity,
    compute_ece,
    kl_per_position,
    entropy_per_position,
    maxprob_per_position,
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
    dataloader : GSM8K DataLoader (already tokenised & collated).
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
    total_nll = 0.0
    total_kl = 0.0
    total_ece_num = 0.0
    total_tokens = 0
    n_batches = 0
    total_seq_length = 0     # sum of non-pad token counts (for avg_seq_length)

    # --- Per-region accumulators ---
    region_acc: Dict[int, Dict[str, float]] = {}
    for rid in (REGION_PROMPT, REGION_REASONING, REGION_ANSWER):
        region_acc[rid] = {
            "entropy": 0.0, "maxprob": 0.0, "nll": 0.0,
            "kl": 0.0, "ece": 0.0, "n_tokens": 0,
        }

    # --- Per-position accumulators ---
    max_pos = cfg["max_length"] - 1
    kl_pos_sum = torch.zeros(max_pos, dtype=torch.float32, device=device)
    kl_pos_cnt = torch.zeros(max_pos, dtype=torch.float32, device=device)
    ent_pos_sum = torch.zeros(max_pos, dtype=torch.float32, device=device)
    ent_pos_cnt = torch.zeros(max_pos, dtype=torch.float32, device=device)
    mp_pos_sum = torch.zeros(max_pos, dtype=torch.float32, device=device)
    mp_pos_cnt = torch.zeros(max_pos, dtype=torch.float32, device=device)

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

        # ---- Average sequence length (non-pad tokens) ----
        total_seq_length += int(batch["attention_mask"].sum().item())

        # ---- Global scalar metrics ----
        batch_entropy = mean_entropy(shift_s, valid_mask).item()
        batch_maxprob = mean_max_probability(shift_s, valid_mask).item()
        batch_nll = mean_nll(shift_s, shift_labels, valid_mask).item()
        batch_ece = compute_ece(shift_s, shift_labels, valid_mask).item()

        total_entropy += batch_entropy * n_valid
        total_maxprob += batch_maxprob * n_valid
        total_nll += batch_nll * n_valid
        total_ece_num += batch_ece * n_valid
        total_tokens += n_valid

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
                racc["nll"] += mean_nll(shift_s, shift_labels, rmask).item() * rn
                racc["ece"] += compute_ece(shift_s, shift_labels, rmask).item() * rn
                racc["n_tokens"] += rn
                if shift_t is not None:
                    racc["kl"] += mean_kl(shift_t, shift_s, rmask).item() * rn

        # ---- Per-position curves ----
        L_cur = shift_s.size(1)
        mask_cnt = valid_mask.float().sum(dim=0)                  # [L_cur]

        ent_curve = entropy_per_position(shift_s, valid_mask)
        ent_pos_sum[:L_cur] += ent_curve * mask_cnt
        ent_pos_cnt[:L_cur] += mask_cnt

        mp_curve = maxprob_per_position(shift_s, valid_mask)
        mp_pos_sum[:L_cur] += mp_curve * mask_cnt
        mp_pos_cnt[:L_cur] += mask_cnt

        if shift_t is not None:
            kl_curve = kl_per_position(shift_t, shift_s, valid_mask)
            kl_pos_sum[:L_cur] += kl_curve * mask_cnt
            kl_pos_cnt[:L_cur] += mask_cnt

        n_batches += 1

    # ---- Aggregate ----
    if total_tokens == 0:
        raise RuntimeError("No valid tokens found in the evaluation set.")

    agg_entropy = total_entropy / total_tokens
    agg_maxprob = total_maxprob / total_tokens
    agg_nll = total_nll / total_tokens
    agg_ppl = math.exp(agg_nll)
    agg_ece = total_ece_num / total_tokens
    agg_kl = (total_kl / total_tokens) if teacher is not None else None

    B_total = n_batches * cfg.get("batch_size", 1)
    avg_seq_len = total_seq_length / max(B_total, 1)

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
        return {
            "entropy": round(racc["entropy"] / rn, 6),
            "maxprob": round(racc["maxprob"] / rn, 6),
            "nll": round(r_nll, 6),
            "ppl": round(math.exp(r_nll), 4),
            "kl": round(racc["kl"] / rn, 6) if teacher is not None else None,
            "ece": round(racc["ece"] / rn, 6),
            "n_tokens": rn,
        }

    results = {
        # Global scalars
        "mean_entropy": round(agg_entropy, 6),
        "mean_maxprob": round(agg_maxprob, 6),
        "mean_nll": round(agg_nll, 6),
        "ppl": round(agg_ppl, 4),
        "mean_kl": round(agg_kl, 6) if agg_kl is not None else None,
        "ece": round(agg_ece, 6),
        # Per-region
        "region_prompt": _aggregate_region(REGION_PROMPT),
        "region_reasoning": _aggregate_region(REGION_REASONING),
        "region_answer": _aggregate_region(REGION_ANSWER),
        # Per-position curves
        "kl_per_position": (
            _curve_to_list(kl_pos_sum, kl_pos_cnt)
            if teacher is not None else None
        ),
        "ent_per_position": _curve_to_list(ent_pos_sum, ent_pos_cnt),
        "mp_per_position": _curve_to_list(mp_pos_sum, mp_pos_cnt),
        # Meta
        "n_tokens_eval": total_tokens,
        "n_batches": n_batches,
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
    for rname, rkey in [("Prompt", "region_prompt"),
                        ("Reasoning", "region_reasoning"),
                        ("Answer", "region_answer")]:
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
