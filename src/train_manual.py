"""
Manual training loop for Knowledge Distillation on a Causal LM.

No HF Trainer – pure PyTorch for full auditability.
Supports three KD modes: ce_only, fkl (Forward KL), rkl (Reverse KL).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

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
# Dataset dispatch
# ------------------------------------------------------------------

def _build_dataloader(cfg: Dict[str, Any], tokenizer):
    """Build the training DataLoader based on config['dataset']."""
    dataset_name = cfg.get("dataset", "gsm8k")

    if dataset_name == "dolly":
        from src.data_dolly import build_dataloader
    elif dataset_name == "gsm8k":
        from src.data_gsm8k import build_dataloader
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Supported: 'dolly', 'gsm8k'."
        )

    micro_n = cfg.get("micro_overfit_n", None)
    return build_dataloader(
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        split="train",
        micro_overfit_n=micro_n,
        shuffle=True,
    )


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(cfg: Dict[str, Any]) -> None:
    """Run the full KD training loop."""

    # ---- Validate config (catch old-style params) ----
    if "lambda_kd" in cfg or "lambda_ce" in cfg:
        raise ValueError(
            "Deprecated config detected: 'lambda_kd' and 'lambda_ce' have been "
            "replaced by 'alpha' and 'kd_mode'. "
            "See configs/dolly_fkl_T2_seed42.yaml for the new format."
        )
    if "num_steps" in cfg and "num_epochs" not in cfg:
        raise ValueError(
            "Deprecated config detected: 'num_steps' has been replaced by "
            "'num_epochs'. Please update your config file."
        )

    # ---- Seed ----
    set_seed(cfg["seed"])

    # ---- Performance: enable TF32 on Ampere+ GPUs ----
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Tokenizer (shared) ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Models ----
    print(f"Loading teacher: {cfg['teacher_name']}  (mode={cfg['teacher_load_mode']})")
    teacher = load_teacher(cfg["teacher_name"], cfg["teacher_load_mode"])
    print(f"Loading student: {cfg['student_name']}  (dtype={cfg['student_dtype']})")
    student = load_student(cfg["student_name"], cfg["student_dtype"])

    # Optional memory-saving toggles controlled by YAML
    enable_gradient_checkpointing = cfg.get("enable_gradient_checkpointing", False)
    free_logits_after_loss = cfg.get("free_logits_after_loss", False)

    if enable_gradient_checkpointing:
        student.gradient_checkpointing_enable()

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
    loader = _build_dataloader(cfg, tokenizer)

    # ---- Hyper-parameters ----
    T = cfg["temperature"]
    alpha = cfg["alpha"]
    kd_mode = cfg["kd_mode"]
    grad_accum_steps = cfg["grad_accum_steps"]
    max_grad_norm = cfg["max_grad_norm"]
    num_epochs = cfg["num_epochs"]
    log_every = cfg["log_every"]
    save_every = cfg.get("save_every", 0)      # 0 = save only at end
    save_dir = cfg.get("save_dir", "checkpoints")
    log_file = cfg.get("log_file", None)
    warmup_ratio = cfg.get("warmup_ratio", 0.1)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # ---- Cosine scheduler with warmup ----
    total_batches = num_epochs * len(loader)
    total_optimizer_steps = total_batches // grad_accum_steps
    warmup_steps = int(warmup_ratio * total_optimizer_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    # Prepare JSONL log
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        log_fh = open(log_file, "w", encoding="utf-8")
    else:
        log_fh = None

    # ---- Training loop ----
    print(f"\nStarting training  –  {num_epochs} epochs, "
          f"{total_optimizer_steps} optimizer steps, "
          f"grad_accum={grad_accum_steps}, batch_size={cfg['batch_size']}")
    print(f"Effective batch = {cfg['batch_size'] * grad_accum_steps}")
    print(f"T={T}, alpha={alpha}, kd_mode={kd_mode}")
    print(f"Scheduler: cosine, warmup={warmup_steps}/{total_optimizer_steps}\n")

    optimizer.zero_grad()
    accum_loss_total = 0.0
    accum_loss_ce = 0.0
    accum_loss_kd = 0.0
    accum_n_tokens = 0
    accum_micro_steps = 0
    micro_count = 0
    global_step = 0
    dtype_logged = False

    for epoch in range(1, num_epochs + 1):
        print(f"--- Epoch {epoch}/{num_epochs} ---")

        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # ---- Teacher forward (frozen, no_grad) ----
            # Skip teacher entirely for ce_only (no KD component)
            if kd_mode != 'ce_only':
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                teacher_logits = teacher_out.logits.detach()
            else:
                # Dummy logits — compute_total_loss won't use them in ce_only
                teacher_logits = torch.empty(0, device=device)

            # ---- Student forward ----
            student_out = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            student_logits = student_out.logits

            # ---- Truncate teacher vocab to match student ----
            if kd_mode != 'ce_only':
                if teacher_logits.size(-1) > student_logits.size(-1):
                    teacher_logits = teacher_logits[..., :student_logits.size(-1)]

                assert teacher_logits.shape[-1] == student_logits.shape[-1], (
                    f"Vocab size mismatch after alignment: "
                    f"teacher {teacher_logits.shape[-1]} != student {student_logits.shape[-1]}"
                )

            # Log dtypes once
            if not dtype_logged:
                print(
                    f"[dtype] teacher_logits={teacher_logits.dtype}, "
                    f"student_logits={student_logits.dtype}"
                )
                dtype_logged = True

            # ---- Loss ----
            loss_total, loss_ce, loss_kd, n_valid = compute_total_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                T=T,
                alpha=alpha,
                kd_mode=kd_mode,
            )

            if free_logits_after_loss:
                # Free logits early to reduce VRAM pressure.
                del teacher_logits, student_logits
                torch.cuda.empty_cache()

            # Scale by grad_accum before backward
            scaled_loss = loss_total / grad_accum_steps
            if scaled_loss.requires_grad:
                scaled_loss.backward()

            # Accumulate for logging
            accum_loss_total += loss_total.item()
            accum_loss_ce += loss_ce.item()
            accum_loss_kd += loss_kd.item()
            accum_n_tokens += n_valid
            accum_micro_steps += 1
            micro_count += 1

            # ---- Optimizer step (when grad_accum micro-steps complete) ----
            if micro_count % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ---- Logging ----
                if global_step % log_every == 0:
                    avg_total = accum_loss_total / max(accum_micro_steps, 1)
                    avg_ce = accum_loss_ce / max(accum_micro_steps, 1)
                    avg_kd = accum_loss_kd / max(accum_micro_steps, 1)
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"[step {global_step:>5d}/{total_optimizer_steps}]  "
                        f"loss={avg_total:.4f}  ce={avg_ce:.4f}  kd={avg_kd:.4f}  "
                        f"tokens={accum_n_tokens}  lr={current_lr:.2e}"
                    )
                    if log_fh is not None:
                        log_fh.write(json.dumps({
                            "step": global_step,
                            "epoch": epoch,
                            "loss_total": round(avg_total, 6),
                            "loss_ce": round(avg_ce, 6),
                            "loss_kd": round(avg_kd, 6),
                            "n_tokens": accum_n_tokens,
                            "lr": round(current_lr, 8),
                        }) + "\n")
                        log_fh.flush()
                    accum_loss_total = 0.0
                    accum_loss_ce = 0.0
                    accum_loss_kd = 0.0
                    accum_n_tokens = 0
                    accum_micro_steps = 0

                # ---- Checkpoint ----
                if save_every > 0 and global_step % save_every == 0:
                    _save_checkpoint(student, tokenizer, save_dir, tag=f"step_{global_step}")

        # ---- Handle trailing micro-steps at end of epoch ----
        remaining = micro_count % grad_accum_steps
        if remaining > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            micro_count = 0  # reset for next epoch

    # ---- Final checkpoint ----
    _save_checkpoint(student, tokenizer, save_dir, tag="final")
    if log_fh is not None:
        log_fh.close()
        print(f"[log] saved → {log_file}")
    print(f"\nTraining complete. Total optimizer steps: {global_step}")
