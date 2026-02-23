"""
Manual training loop for Knowledge Distillation on a Causal LM.

No HF Trainer – pure PyTorch for full auditability.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data_gsm8k import build_dataloader
from src.losses_kd import compute_total_loss
from src.utils_seed import set_seed


# ------------------------------------------------------------------
# Model loading helpers
# ------------------------------------------------------------------

def load_teacher(name: str, load_mode: str) -> AutoModelForCausalLM:
    """Load the teacher model according to ``load_mode`` (4bit | 8bit | bf16).

    The teacher is always frozen and set to eval.
    """
    if load_mode == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    elif load_mode == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    elif load_mode == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unknown teacher_load_mode: {load_mode!r}")

    # Freeze teacher
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def load_student(name: str, dtype_str: str) -> AutoModelForCausalLM:
    """Load the student model in the requested dtype."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.train()
    return model


# ------------------------------------------------------------------
# Checkpoint helper
# ------------------------------------------------------------------

def _save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_dir: str,
    tag: str,
) -> None:
    """Save student model + tokenizer to ``save_dir/tag``."""
    path = os.path.join(save_dir, tag)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[ckpt] saved → {path}")


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(cfg: Dict[str, Any]) -> None:
    """Run the full KD training loop."""

    # ---- Seed ----
    set_seed(cfg["seed"])

    # ---- Tokenizer (shared) ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Models ----
    print(f"Loading teacher: {cfg['teacher_name']}  (mode={cfg['teacher_load_mode']})")
    teacher = load_teacher(cfg["teacher_name"], cfg["teacher_load_mode"])
    print(f"Loading student: {cfg['student_name']}  (dtype={cfg['student_dtype']})")
    student = load_student(cfg["student_name"], cfg["student_dtype"])

    device = next(student.parameters()).device

    # ---- Vocab-size check ----
    V_teacher = teacher.config.vocab_size
    V_student = student.config.vocab_size
    if V_teacher != V_student:
        print(
            f"[vocab] teacher={V_teacher}, student={V_student} "
            f"→ teacher logits will be truncated to {V_student}"
        )
    # Verify that shared token ids map to the same tokens
    teacher_tok = AutoTokenizer.from_pretrained(cfg["teacher_name"])
    for i in range(min(V_teacher, V_student)):
        t_tok = teacher_tok.convert_ids_to_tokens(i)
        s_tok = tokenizer.convert_ids_to_tokens(i)
        assert t_tok == s_tok, (
            f"Token id {i} mismatch: teacher='{t_tok}' vs student='{s_tok}'. "
            f"Cannot safely truncate teacher logits."
        )
    print(f"[vocab] token id consistency verified for {min(V_teacher, V_student)} tokens ✓")
    del teacher_tok

    # ---- Data ----
    micro_n = cfg.get("micro_overfit_n", None)
    loader = build_dataloader(
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        split="train",
        micro_overfit_n=micro_n,
        shuffle=True,
    )
    data_iter = iter(loader)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # ---- Hyper-parameters ----
    T = cfg["temperature"]
    lambda_kd = cfg["lambda_kd"]
    lambda_ce = cfg["lambda_ce"]
    grad_accum_steps = cfg["grad_accum_steps"]
    max_grad_norm = cfg["max_grad_norm"]
    num_steps = cfg["num_steps"]
    log_every = cfg["log_every"]
    save_every = cfg.get("save_every", 0)      # 0 = save only at end
    save_dir = cfg.get("save_dir", "checkpoints")
    log_file = cfg.get("log_file", None)          # optional JSONL log path

    # Prepare JSONL log
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        log_fh = open(log_file, "w", encoding="utf-8")
    else:
        log_fh = None

    # ---- Training loop ----
    print(f"\nStarting training  –  {num_steps} optimiser steps, "
          f"grad_accum={grad_accum_steps}, batch_size={cfg['batch_size']}")
    print(f"Effective batch = {cfg['batch_size'] * grad_accum_steps}")
    print(f"T={T}, λ_KD={lambda_kd}, λ_CE={lambda_ce}\n")

    optimizer.zero_grad()
    accum_loss_total = 0.0
    accum_loss_ce = 0.0
    accum_loss_kd = 0.0
    accum_n_tokens = 0

    for step in range(1, num_steps + 1):
        for micro_step in range(grad_accum_steps):
            # Fetch batch (cycle if dataset is small)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            # ---- Teacher forward (frozen, no_grad) ----
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            teacher_logits = teacher_out.logits.detach()

            # ---- Student forward ----
            student_out = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            student_logits = student_out.logits

            # ---- Truncate teacher vocab to match student ----
            if teacher_logits.size(-1) > student_logits.size(-1):
                teacher_logits = teacher_logits[..., :student_logits.size(-1)]

            assert teacher_logits.shape[-1] == student_logits.shape[-1], (
                f"Vocab size mismatch after alignment: "
                f"teacher {teacher_logits.shape[-1]} != student {student_logits.shape[-1]}"
            )
            # Log dtypes once (first micro-step of first optimiser step)
            if step == 1 and micro_step == 0:
                print(
                    f"[dtype] teacher_logits={teacher_logits.dtype}, "
                    f"student_logits={student_logits.dtype}"
                )

            # ---- Loss ----
            loss_total, loss_ce, loss_kd, n_valid = compute_total_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                T=T,
                lambda_kd=lambda_kd,
                lambda_ce=lambda_ce,
            )

            # Scale by grad_accum before backward
            scaled_loss = loss_total / grad_accum_steps
            scaled_loss.backward()

            # Accumulate for logging
            accum_loss_total += loss_total.item()
            accum_loss_ce += loss_ce.item()
            accum_loss_kd += loss_kd.item()
            accum_n_tokens += n_valid

        # ---- Gradient clipping ----
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)

        # ---- Optimiser step ----
        optimizer.step()
        optimizer.zero_grad()

        # ---- Logging ----
        if step % log_every == 0:
            avg_total = accum_loss_total / grad_accum_steps
            avg_ce = accum_loss_ce / grad_accum_steps
            avg_kd = accum_loss_kd / grad_accum_steps
            print(
                f"[step {step:>5d}/{num_steps}]  "
                f"loss={avg_total:.4f}  ce={avg_ce:.4f}  kd={avg_kd:.4f}  "
                f"tokens={accum_n_tokens}"
            )
            if log_fh is not None:
                log_fh.write(json.dumps({
                    "step": step,
                    "loss_total": round(avg_total, 6),
                    "loss_ce": round(avg_ce, 6),
                    "loss_kd": round(avg_kd, 6),
                    "n_tokens": accum_n_tokens,
                }) + "\n")
                log_fh.flush()
            accum_loss_total = 0.0
            accum_loss_ce = 0.0
            accum_loss_kd = 0.0
            accum_n_tokens = 0

        # ---- Checkpoint ----
        if save_every > 0 and step % save_every == 0:
            _save_checkpoint(student, tokenizer, save_dir, tag=f"step_{step}")

    # ---- Final checkpoint ----
    _save_checkpoint(student, tokenizer, save_dir, tag="final")
    if log_fh is not None:
        log_fh.close()
        print(f"[log] saved → {log_file}")
    print("\nTraining complete.")
