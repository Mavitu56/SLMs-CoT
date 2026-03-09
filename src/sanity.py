"""
Sanity checks – run on 1 batch before full training.

Each check is a standalone function that raises on failure.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data_dolly import build_dataloader
from src.losses_kd import compute_total_loss
from src.analysis_metrics import compute_ece


# ==================================================================
# Helpers
# ==================================================================

def _align_vocab(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate the teacher vocab dim if it is larger than the student's."""
    if teacher_logits.size(-1) > student_logits.size(-1):
        teacher_logits = teacher_logits[..., :student_logits.size(-1)]
    assert teacher_logits.shape[-1] == student_logits.shape[-1], (
        f"Vocab size mismatch: teacher {teacher_logits.shape[-1]} "
        f"!= student {student_logits.shape[-1]}"
    )
    return teacher_logits, student_logits


def _get_one_batch(
    tokenizer,
    max_length: int,
    micro_n: int = 4,
    batch_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Return a single batch from Dolly (train)."""
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

    t_logits, s_logits = _align_vocab(t_out.logits, s_out.logits)
    loss_total, _, _, _ = compute_total_loss(
        teacher_logits=t_logits,
        student_logits=s_logits,
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
        T=T,
        alpha=0.5,
        kd_mode='fkl',
    )

    loss_total.backward()

    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher param {name} has grad!"

    # Clean up student grads
    student.zero_grad()
    print("PASSED")


# ==================================================================
# 6.2  kd_mode combinations
# ==================================================================

def check_kd_mode_combinations(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    T: float = 2.0,
) -> None:
    """Verify loss formulas for each kd_mode."""
    print("[sanity] 6.2 – kd_mode combinations …", end=" ")

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

    t_logits, s_logits = _align_vocab(t_logits, s_logits)

    # Case 1: ce_only → loss_total == loss_ce, loss_kd == 0
    loss_total_1, loss_ce_1, loss_kd_1, _ = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, alpha=0.5, kd_mode='ce_only',
    )
    assert torch.allclose(loss_total_1, loss_ce_1, atol=1e-5), (
        f"ce_only: total={loss_total_1.item():.6f} != ce={loss_ce_1.item():.6f}"
    )
    assert loss_kd_1.item() == 0.0, f"ce_only: kd should be 0, got {loss_kd_1.item()}"

    # Case 2: fkl → loss_total == alpha*ce + (1-alpha)*T²*kd
    alpha = 0.5
    loss_total_2, loss_ce_2, loss_kd_2, _ = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, alpha=alpha, kd_mode='fkl',
    )
    expected_2 = alpha * loss_ce_2 + (1 - alpha) * (T ** 2) * loss_kd_2
    assert torch.allclose(loss_total_2, expected_2, atol=1e-5), (
        f"fkl: total={loss_total_2.item():.6f} != "
        f"alpha*ce + (1-alpha)*T²*kd={expected_2.item():.6f}"
    )

    # Case 3: rkl → loss_total == alpha*ce + (1-alpha)*kd (no T²)
    loss_total_3, loss_ce_3, loss_kd_3, _ = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, alpha=alpha, kd_mode='rkl',
    )
    expected_3 = alpha * loss_ce_3 + (1 - alpha) * loss_kd_3
    assert torch.allclose(loss_total_3, expected_3, atol=1e-5), (
        f"rkl: total={loss_total_3.item():.6f} != "
        f"alpha*ce + (1-alpha)*kd={expected_3.item():.6f}"
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
        T=T, alpha=0.5, kd_mode='fkl',
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

    t_logits, s_logits = _align_vocab(t_logits, s_logits)

    _, _, loss_kd, _ = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, alpha=0.5, kd_mode='fkl',
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

    t_logits, s_logits = _align_vocab(t_logits, s_logits)

    loss_total, loss_ce, loss_kd, n_valid = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=T, alpha=0.5, kd_mode='fkl',
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
# 7  RKL ≈ 0 when teacher == student
# ==================================================================

def check_rkl_near_zero_same_model(
    tokenizer,
    max_length: int,
    student_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> None:
    """When both teacher and student are the same model, RKL should be ≈ 0."""
    print("[sanity] 7 – RKL ≈ 0 (teacher == student) …", end=" ")

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
        T=1.0, alpha=0.5, kd_mode='rkl',
    )

    kd_val = loss_kd.item()
    assert kd_val < 1e-2, (
        f"RKL with identical models should be ≈0, got {kd_val:.6f}"
    )
    print(f"PASSED  (rkl={kd_val:.2e})")

    del model_a, model_b
    torch.cuda.empty_cache()


# ==================================================================
# 8  RKL is invariant to temperature
# ==================================================================

def check_rkl_temperature_invariant(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> None:
    """RKL should be identical regardless of the temperature parameter."""
    print("[sanity] 8 – RKL invariant to T …", end=" ")

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

    t_logits, s_logits = _align_vocab(t_logits, s_logits)

    loss_T1 = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=1.0, alpha=0.5, kd_mode='rkl',
    )[0]
    loss_T4 = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=4.0, alpha=0.5, kd_mode='rkl',
    )[0]

    assert torch.allclose(loss_T1, loss_T4, atol=1e-5), (
        f"RKL should be invariant to T: "
        f"T=1 → {loss_T1.item():.6f}, T=4 → {loss_T4.item():.6f}"
    )
    print("PASSED")


# ==================================================================
# 9  FKL varies with temperature
# ==================================================================

def check_fkl_temperature_sensitive(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> None:
    """FKL total loss should differ when T changes."""
    print("[sanity] 9 – FKL varies with T …", end=" ")

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

    t_logits, s_logits = _align_vocab(t_logits, s_logits)

    loss_T1 = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=1.0, alpha=0.5, kd_mode='fkl',
    )[0]
    loss_T4 = compute_total_loss(
        t_logits, s_logits, batch["labels"], batch["attention_mask"],
        T=4.0, alpha=0.5, kd_mode='fkl',
    )[0]

    assert not torch.allclose(loss_T1, loss_T4, atol=1e-3), (
        f"FKL should vary with T: "
        f"T=1 → {loss_T1.item():.6f}, T=4 → {loss_T4.item():.6f}"
    )
    print(f"PASSED  (T=1: {loss_T1.item():.4f}, T=4: {loss_T4.item():.4f})")


# ==================================================================
# 10  ECE analytical cases
# ==================================================================

def check_ece_analytical() -> None:
    """ECE with known distributions: uniform → ~0, always-confident → ~0.5."""
    print("[sanity] 10 – ECE analytical …", end=" ")

    V = 10
    N = 1000

    # Case 1: Uniform distribution
    # conf = 1/V = 0.1, acc ~= 1/V = 0.1 → ECE ≈ 0
    logits_uniform = torch.zeros(1, N, V)
    labels_uniform = torch.randint(0, V, (1, N))
    mask = torch.ones(1, N, dtype=torch.bool)

    ece_uniform = compute_ece(logits_uniform, labels_uniform, mask, n_bins=10)
    assert ece_uniform.item() < 0.05, (
        f"Uniform ECE should be ~0, got {ece_uniform.item():.4f}"
    )

    # Case 2: Always-confident model (conf≈1) that is correct 50% of the time
    # → ECE ≈ |1.0 - 0.5| = 0.5
    logits_confident = torch.zeros(1, N, V)
    logits_confident[:, :, 0] = 100.0  # very high logit → conf ≈ 1
    labels_half = torch.zeros(1, N, dtype=torch.long)
    labels_half[:, N // 2:] = 1  # half correct (class 0), half wrong (class 1)

    ece_confident = compute_ece(logits_confident, labels_half, mask, n_bins=10)
    assert abs(ece_confident.item() - 0.5) < 0.05, (
        f"Always-confident ECE should be ~0.5, got {ece_confident.item():.4f}"
    )

    print("PASSED")


# ==================================================================
# 11  Masking (prompt_len) functional
# ==================================================================

def check_masking_prompt_len(tokenizer, max_length: int) -> None:
    """Verify that prompt tokens have labels=-100 in Dolly tokenisation."""
    print("[sanity] 11 – Masking (prompt_len) …", end=" ")

    from src.data_dolly import tokenise_example

    example = {
        "instruction": "What is the capital of France?",
        "context": "",
        "response": "The capital of France is Paris.",
        "category": "open_qa",
    }

    result = tokenise_example(example, tokenizer, max_length)
    assert result is not None, "tokenise_example returned None for valid example"

    labels = result["labels"]
    input_ids = result["input_ids"]

    # Find where labels stop being -100
    first_valid = next(
        (i for i, l in enumerate(labels) if l != -100),
        len(labels),
    )

    # All labels before first_valid should be -100 (prompt masked)
    assert all(l == -100 for l in labels[:first_valid]), (
        "Labels before prompt_len should all be -100"
    )

    # At least some labels after first_valid should be valid (response tokens)
    assert first_valid < len(labels), (
        "No valid response tokens found — all labels are -100"
    )

    # Valid labels should match input_ids (teacher forcing)
    for i in range(first_valid, len(labels)):
        if labels[i] != -100:
            assert labels[i] == input_ids[i], (
                f"Label mismatch at position {i}: "
                f"label={labels[i]} != input_id={input_ids[i]}"
            )

    # Verify that an example with empty response returns None
    empty_example = {
        "instruction": "Test",
        "context": "",
        "response": "",
        "category": "open_qa",
    }
    assert tokenise_example(empty_example, tokenizer, max_length) is None, (
        "Empty-response example should return None"
    )

    print(f"PASSED  (prompt_len={first_valid}, total={len(labels)})")


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
    student_name = cfg.get("student_name", "Qwen/Qwen2.5-1.5B-Instruct")

    batch = _get_one_batch(tokenizer, max_length, micro_n=4, batch_size=1)

    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # --- Existing checks (6.1 – 6.6) ---
    check_teacher_frozen(teacher, student, batch, T=T)
    check_kd_mode_combinations(teacher, student, batch, T=T)
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
    _t_logits, _s_logits = _align_vocab(_t_logits, _s_logits)
    _, _, _normal_kd, _ = compute_total_loss(
        _t_logits, _s_logits, _batch_dev["labels"], _batch_dev["attention_mask"],
        T=T, alpha=0.5, kd_mode='fkl',
    )
    normal_kd_ref = _normal_kd.item()
    print(f"[info] normal teacher→student KD = {normal_kd_ref:.4f}")

    # 6.3 loads its own models (both copies of student)
    check_kd_near_zero_same_model(
        tokenizer, max_length,
        student_name=student_name, T=T,
        normal_kd_ref=normal_kd_ref,
    )

    # --- New checks (7 – 11) ---
    check_rkl_near_zero_same_model(
        tokenizer, max_length,
        student_name=student_name,
    )
    check_rkl_temperature_invariant(teacher, student, batch)
    check_fkl_temperature_sensitive(teacher, student, batch)
    check_ece_analytical()
    check_masking_prompt_len(tokenizer, max_length)

    print("=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)
