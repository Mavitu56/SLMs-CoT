from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import json

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import get_scheduler

from config import (
    EvidenceBasedConfig,
    get_schedule_value,
    safe_model_to,
    set_seed,
)
from prompts import build_cot_prompt


def _resolve_amp_device_type(device: torch.device) -> str:
    try:
        dtype = getattr(device, "type", None)
        return str(dtype) if dtype else "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cuda" if torch.cuda.is_available() else "cpu"


def make_grad_scaler(device: torch.device):
    device_type = _resolve_amp_device_type(device)
    enabled = device_type == "cuda"
    try:
        from torch import amp as torch_amp

        try:
            return torch_amp.GradScaler(device_type=device_type, enabled=enabled)
        except TypeError:
            return torch_amp.GradScaler(device_type, enabled=enabled)
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler

        return CudaGradScaler(enabled=enabled)


def autocast_ctx(device: torch.device):
    device_type = _resolve_amp_device_type(device)
    enabled = device_type == "cuda"
    try:
        from torch import amp as torch_amp

        return torch_amp.autocast(device_type=device_type, enabled=enabled)
    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast

        return cuda_autocast(enabled=enabled)


def preprocess_and_tokenize(raw_dataset, tokenizer, max_length: int):
    """Tokenize training inputs and build labels.

    Scientific validity fix (confounding): when `prompt`/`completion` are
    available, we mask prompt tokens so that CE/KD apply only to completion.
    This keeps the traditional and reasoning-aware conditions aligned.
    """

    has_prompt = False
    try:
        cols = getattr(raw_dataset, "column_names", []) or []
        has_prompt = "prompt" in cols and "completion" in cols
    except Exception:
        has_prompt = False

    def fn(examples):
        texts = examples.get("text")
        toks = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        labels = [ids.copy() for ids in toks["input_ids"]]
        for i, attn in enumerate(toks["attention_mask"]):
            for j, mask in enumerate(attn):
                if mask == 0:
                    labels[i][j] = -100

        if has_prompt:
            prompts = examples.get("prompt")
            ptoks = tokenizer(
                prompts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            prompt_lens = []
            for attn in ptoks["attention_mask"]:
                prompt_lens.append(int(sum(attn)))
            for i, plen in enumerate(prompt_lens):
                plen = max(0, min(int(plen), max_length))
                for j in range(plen):
                    labels[i][j] = -100

        toks["labels"] = labels
        return toks

    tokenized = raw_dataset.map(fn, batched=True, batch_size=1024, remove_columns=getattr(raw_dataset, "column_names", None))
    tokenized.set_format(type="torch")
    return tokenized


def build_reasoning_full_sequences_from_cot(
    cot_path: str,
    max_records: Optional[int] = None,
    *,
    granularity_level: int = 0,
    post_cot: bool = False,
    post_cot_use_ig: bool = False,
) -> List[str]:
    """Build prompt+completion sequences from a teacher CoT cache.

    Scientific validity fix: reasoning-mode logits cache (if enabled) must be
    computed on the same input sequences used for training.
    """

    import json

    seqs: List[str] = []
    with open(cot_path, "r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)

            # Multi-level cache: expand across cached granularity levels.
            prompt_levels = rec.get("prompt_levels")
            reasoning_levels = rec.get("teacher_reasoning_levels")
            answer_levels = rec.get("teacher_answer_levels")

            if isinstance(prompt_levels, dict) and isinstance(reasoning_levels, dict):
                levels = rec.get("granularity_levels")
                if not isinstance(levels, list) or not levels:
                    try:
                        levels = sorted({int(k) for k in prompt_levels.keys()})
                    except Exception:
                        levels = []
                for lvl in levels:
                    k = str(int(lvl))
                    prompt = str(prompt_levels.get(k) or "").strip()
                    if not prompt:
                        continue
                    reasoning = str(reasoning_levels.get(k) or "").strip()
                    if not reasoning:
                        continue
                    answer = str((answer_levels or {}).get(k) or rec.get("teacher_answer", "") or "").strip()

                    if bool(post_cot):
                        if bool(post_cot_use_ig) and isinstance(rec.get("teacher_reasoning_ig"), str) and rec.get("teacher_reasoning_ig").strip():
                            reasoning_use = str(rec.get("teacher_reasoning_ig") or "").strip()
                        else:
                            reasoning_use = reasoning
                        teacher_full = (answer + "\n### REASONING:\n" + reasoning_use)
                    else:
                        teacher_full = (reasoning + "\n### FINAL_ANSWER: " + answer)

                    seqs.append(prompt + teacher_full)
                    if max_records is not None and len(seqs) >= int(max_records):
                        break
                if max_records is not None and len(seqs) >= int(max_records):
                    break
                continue

            # Single-level legacy.
            prompt = rec.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                prompt, _ = build_cot_prompt(
                    str(rec.get("question", rec.get("text", ""))),
                    granularity_level=int(granularity_level or 0),
                    post_cot=bool(post_cot),
                )

            if bool(post_cot):
                reasoning = rec.get("teacher_reasoning", "").strip()
                if bool(post_cot_use_ig) and isinstance(rec.get("teacher_reasoning_ig"), str) and rec.get("teacher_reasoning_ig").strip():
                    reasoning = str(rec.get("teacher_reasoning_ig") or "").strip()
                teacher_full = (rec.get("teacher_answer", "").strip() + "\n### REASONING:\n" + reasoning)
            else:
                teacher_full = (rec.get("teacher_reasoning", "").strip() + "\n### FINAL_ANSWER: " + rec.get("teacher_answer", ""))
            seqs.append(prompt + teacher_full)
            if max_records is not None and len(seqs) >= int(max_records):
                break
    return seqs


def sanitize_labels_for_ce(labels: torch.Tensor, vocab_size: int, ignore_index: int = -100):
    invalid_high = labels >= vocab_size
    invalid_low = (labels < 0) & (labels != ignore_index)
    invalid_total = int(invalid_high.sum().item() + invalid_low.sum().item())
    if invalid_total:
        labels = labels.clone()
        if invalid_high.any():
            labels[invalid_high] = vocab_size - 1
        if invalid_low.any():
            labels[invalid_low] = ignore_index
    return labels, invalid_total


class TraditionalKDDistiller:
    def __init__(self, config: EvidenceBasedConfig, cache_dir: Optional[str] = None):
        self.config = config
        self.cache_dir = cache_dir

    def _ensure_vocab_alignment(self, model, tokenizer):
        tok_vocab = len(tokenizer)
        model_vocab = model.get_input_embeddings().weight.size(0)
        # Safety: only expand embeddings. Shrinking can break HF models that
        # intentionally reserve extra rows and can desync output heads.
        if tok_vocab > model_vocab:
            print(f" Resizing student embeddings (expand): tokenizer={tok_vocab}, model={model_vocab}")
            model.resize_token_embeddings(tok_vocab)
            model = safe_model_to(model, self.config.device)
        return model

    def _align_teacher_logits(self, teacher_logits: torch.Tensor, student_vocab: int) -> torch.Tensor:
        t_vocab = teacher_logits.size(-1)
        if t_vocab == student_vocab:
            return teacher_logits
        if t_vocab > student_vocab:
            return teacher_logits[..., :student_vocab]
        pad = teacher_logits.new_full((*teacher_logits.shape[:-1], student_vocab - t_vocab), -1e9)
        return torch.cat([teacher_logits, pad], dim=-1)

    def distill(self, student_model, teacher_model, student_tokenizer, raw_dataset, seed: int = 42, use_cache: bool = True):
        set_seed(seed)
        device = self.config.device
        cache_available = bool(use_cache and self.cache_dir and os.path.isdir(self.cache_dir))
        if not cache_available and teacher_model is None:
            raise ValueError("teacher_model precisa ser fornecido quando no h cache de logits.")

        # Optional checkpointing + cache-alignment diagnostics (env-driven).
        # These help pinpoint when training starts to drift and whether cached
        # logits correspond to the same tokenized inputs.
        from pathlib import Path

        ckpt_root = os.environ.get("SLM_KD_CKPT_DIR")
        ckpt_root_path = Path(ckpt_root) if ckpt_root else None

        def _env_flag(name: str, default: str = "0") -> bool:
            v = os.environ.get(name, default)
            return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

        def _env_float(name: str, default: float) -> float:
            v = os.environ.get(name)
            if v is None:
                return float(default)
            try:
                return float(str(v).strip())
            except Exception:
                return float(default)

        def _env_int(name: str, default: int) -> int:
            v = os.environ.get(name)
            if v is None:
                return int(default)
            try:
                return int(str(v).strip())
            except Exception:
                return int(default)

        # Debug/stability knobs (env-driven to avoid changing CLI/Config surface).
        debug_nan = _env_flag("SLM_KD_DEBUG_NAN", "1")
        log_every = max(1, _env_int("SLM_KD_LOG_EVERY", 50))
        skip_nonfinite_batch = _env_flag("SLM_KD_SKIP_NONFINITE_BATCH", "1")
        save_every_epoch = _env_flag("SLM_KD_SAVE_EVERY_EPOCH", "0")
        strict_cache_match = _env_flag("SLM_KD_STRICT_CACHE_MATCH", "0")
        clip_grad_norm = float(self.config.kd_params.get("clip_grad_norm", 1.0))
        clip_grad_norm = float(os.environ.get("SLM_KD_CLIP_GRAD_NORM", clip_grad_norm))
        max_logit_abs = _env_float("SLM_KD_MAX_LOGIT_ABS", 100.0)
        apply_logit_sanitize = _env_flag("SLM_KD_SANITIZE_LOGITS", "1")

        def _sanitize_logits(x: torch.Tensor) -> torch.Tensor:
            if not apply_logit_sanitize:
                return x
            # Replace NaN/inf to keep softmax/log_softmax stable.
            x = torch.nan_to_num(x, nan=0.0, posinf=max_logit_abs, neginf=-max_logit_abs)
            if max_logit_abs > 0:
                x = x.clamp(min=-max_logit_abs, max=max_logit_abs)
            return x

        def _has_nonfinite(x: torch.Tensor) -> bool:
            try:
                return bool((~torch.isfinite(x)).any().item())
            except Exception:
                return True

        tokenized = preprocess_and_tokenize(raw_dataset, student_tokenizer, max_length=self.config.max_length)
        student_model = self._ensure_vocab_alignment(student_model, student_tokenizer)

        # Training stability + memory: disable KV cache and enable gradient checkpointing.
        try:
            student_model.config.use_cache = False
        except Exception:
            pass
        try:
            student_model.gradient_checkpointing_enable()
        except Exception:
            pass

        # If the base model is 4-bit/8-bit, use PEFT k-bit preparation (QLoRA style).
        if getattr(student_model, "is_loaded_in_4bit", False) or getattr(student_model, "is_loaded_in_8bit", False):
            from peft import prepare_model_for_kbit_training

            student_model = prepare_model_for_kbit_training(student_model)

        lora_cfg = LoraConfig(
            r=self.config.kd_params["lora_rank"],
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        student_model = get_peft_model(student_model, lora_cfg)
        student_model.train()
        student_model = safe_model_to(student_model, device)

        if teacher_model is not None:
            teacher_model.eval()
            teacher_model = safe_model_to(teacher_model, device)
            try:
                teacher_model.config.use_cache = False
            except Exception:
                pass

        batch_size = int(self.config.kd_params.get("batch_size", 2))
        grad_accum_steps = int(self.config.kd_params.get("grad_accum_steps", 1))
        num_workers = int(self.config.kd_params.get("dataloader_num_workers", 0))
        shuffle_data = not cache_available
        dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=shuffle_data, pin_memory=True, num_workers=num_workers)

        teacher_logits_cache = None
        shard_iter = None
        if cache_available:
            from pathlib import Path

            teacher_logits_cache = sorted(Path(self.cache_dir).glob("teacher_logits_shard_*.pt"))
            shard_iter = iter(teacher_logits_cache)
            print(f" Cache de logits: {len(teacher_logits_cache)} shards")

        num_epochs = self.config.kd_params["epochs"]
        num_training_steps = num_epochs * len(dataloader)
        warmup_steps = int(0.1 * num_training_steps)

        optimizer = torch.optim.AdamW(student_model.parameters(), lr=self.config.kd_params["learning_rates"]["kd"], weight_decay=0.01)
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        # AMP: prefer torch.amp (new API) with backward-compatible fallback.
        scaler = make_grad_scaler(device)

        temperature_schedule = self.config.kd_params.get("temperature_schedule") or [3.0]
        alpha_schedule = self.config.kd_params.get("alpha_schedule") or [0.7]

        metrics = {"losses": [], "kd_losses": [], "ce_losses": []}

        nonfinite_batches = 0
        nonfinite_teacher_shards = 0
        nonfinite_student_logits = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            temperature = get_schedule_value(temperature_schedule, epoch, default=3.0)
            alpha = get_schedule_value(alpha_schedule, epoch, default=0.7)

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"KD Tradicional Epoch {epoch}")):
                inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
                labels = batch["labels"].to(device)
                vocab_size = student_model.get_input_embeddings().weight.size(0)
                labels = labels.long().clamp(min=-100, max=vocab_size - 1)
                labels, _san = sanitize_labels_for_ce(labels, vocab_size)

                if shard_iter is not None:
                    try:
                        shard_path = next(shard_iter)
                    except StopIteration:
                        shard_iter = iter(teacher_logits_cache)
                        shard_path = next(shard_iter)
                    data = torch.load(shard_path, map_location="cpu")

                    # Cache/input alignment checks: verifies the cached logits were
                    # produced for the exact same token IDs and attention mask.
                    # If this fails, logits-KD becomes scientifically invalid.
                    try:
                        cached_ids = data.get("input_ids")
                        cached_mask = data.get("attention_mask")
                        if cached_ids is not None and cached_mask is not None:
                            cur_ids = inputs.get("input_ids").detach().to("cpu")
                            cur_mask = inputs.get("attention_mask").detach().to("cpu")
                            same_ids = bool(torch.equal(cached_ids, cur_ids))
                            same_mask = bool(torch.equal(cached_mask, cur_mask))
                            if not (same_ids and same_mask):
                                msg = (
                                    " (warn) Cache de logits parece desalinhado com o batch atual. "
                                    f"epoch={epoch} batch={batch_idx} shard={Path(shard_path).name} "
                                    f"same_input_ids={same_ids} same_attention_mask={same_mask}"
                                )
                                print(msg)
                                if strict_cache_match:
                                    raise RuntimeError(msg)
                    except Exception as exc:
                        if strict_cache_match:
                            raise
                        if debug_nan:
                            print(f" (warn) Falha ao checar alinhamento do cache: {exc}")

                    # Stability: fp16 logits + softmax can be numerically fragile.
                    teacher_logits = data["logits"].to(device=device, dtype=torch.float32)

                    # Sanity: cached logits batch dimension must match current batch.
                    try:
                        if int(teacher_logits.size(0)) != int(inputs["input_ids"].size(0)):
                            raise RuntimeError(
                                "Batch size mismatch entre cache e dataloader. "
                                f"cache_bs={int(teacher_logits.size(0))} batch_bs={int(inputs['input_ids'].size(0))} "
                                f"shard={Path(shard_path).name}. "
                                "Isso normalmente indica train_limit diferente, batch_size diferente, ou cache incorreto."
                            )
                    except Exception:
                        raise

                    teacher_logits = self._align_teacher_logits(teacher_logits, vocab_size)
                    if _has_nonfinite(teacher_logits):
                        nonfinite_teacher_shards += 1
                        if debug_nan:
                            print(f" (warn) teacher_logits não-finitos no shard: {shard_path}")
                        teacher_logits = _sanitize_logits(teacher_logits)
                else:
                    with torch.no_grad():
                        t_out = teacher_model(**inputs)
                        teacher_logits = self._align_teacher_logits(t_out.logits, vocab_size)
                        teacher_logits = teacher_logits.to(dtype=torch.float32)
                        if _has_nonfinite(teacher_logits):
                            nonfinite_teacher_shards += 1
                            if debug_nan:
                                print(" (warn) teacher_model produziu logits não-finitos (on-the-fly)")
                            teacher_logits = _sanitize_logits(teacher_logits)

                # Forward can be AMP, but compute losses in fp32 outside autocast.
                with autocast_ctx(device):
                    s_logits = student_model(**inputs).logits

                s_logits_f32 = s_logits.to(dtype=torch.float32)
                if _has_nonfinite(s_logits_f32):
                    nonfinite_student_logits += 1
                    if debug_nan:
                        print(f" (warn) student logits não-finitos em epoch={epoch} batch={batch_idx}")
                    s_logits_f32 = _sanitize_logits(s_logits_f32)

                teacher_logits = _sanitize_logits(teacher_logits)

                vocab_size = s_logits_f32.size(-1)
                ce_loss = F.cross_entropy(s_logits_f32.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)

                # KD math in fp32 for stability (especially with large vocabs).
                t_probs = F.softmax(teacher_logits / float(temperature), dim=-1)
                s_logp = F.log_softmax(s_logits_f32 / float(temperature), dim=-1)
                token_kl = F.kl_div(s_logp, t_probs, reduction="none")
                mask = (labels != -100).unsqueeze(-1).float()
                kd_loss = (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
                kd_loss *= temperature**2
                loss = alpha * kd_loss + (1.0 - alpha) * ce_loss

                if _has_nonfinite(loss) or _has_nonfinite(kd_loss) or _has_nonfinite(ce_loss):
                    nonfinite_batches += 1
                    if debug_nan:
                        lr_now = None
                        try:
                            lr_now = float(lr_scheduler.get_last_lr()[0])
                        except Exception:
                            pass
                        print(
                            " (warn) loss não-finita; pulando batch. "
                            f"epoch={epoch} batch={batch_idx} lr={lr_now} "
                            f"loss={float(loss.detach().cpu().item()) if torch.isfinite(loss).all() else 'nonfinite'} "
                            f"kd={float(kd_loss.detach().cpu().item()) if torch.isfinite(kd_loss).all() else 'nonfinite'} "
                            f"ce={float(ce_loss.detach().cpu().item()) if torch.isfinite(ce_loss).all() else 'nonfinite'}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if not skip_nonfinite_batch:
                        raise RuntimeError("Encontrado loss/logits não-finitos durante KD. Veja logs acima.")
                    continue

                if grad_accum_steps > 1:
                    loss = loss / float(grad_accum_steps)

                scaler.scale(loss).backward()

                do_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
                if do_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), float(clip_grad_norm))
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm_val = float(grad_norm.detach().cpu().item())
                    else:
                        grad_norm_val = float(grad_norm)
                    if (not (grad_norm_val == grad_norm_val)) or (grad_norm_val == float("inf")):
                        nonfinite_batches += 1
                        if debug_nan:
                            print(f" (warn) grad_norm não-finito; zerando grad e pulando step. epoch={epoch} batch={batch_idx}")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Store *unscaled* loss values for monitoring.
                try:
                    metrics["losses"].append(float((loss.detach().cpu().item() * grad_accum_steps) if grad_accum_steps > 1 else loss.detach().cpu().item()))
                    metrics["kd_losses"].append(float(kd_loss.detach().cpu().item()))
                    metrics["ce_losses"].append(float(ce_loss.detach().cpu().item()))
                except Exception:
                    pass
                # Track unscaled loss for reporting.
                epoch_loss += float((loss.detach().cpu().item() * grad_accum_steps) if grad_accum_steps > 1 else loss.detach().cpu().item())

                if debug_nan and ((batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(dataloader)):
                    lr_now = None
                    try:
                        lr_now = float(lr_scheduler.get_last_lr()[0])
                    except Exception:
                        pass
                    loss_val = metrics["losses"][-1] if metrics["losses"] else float("nan")
                    kd_val = metrics["kd_losses"][-1] if metrics["kd_losses"] else float("nan")
                    ce_val = metrics["ce_losses"][-1] if metrics["ce_losses"] else float("nan")
                    with torch.no_grad():
                        s_abs = float(s_logits_f32.detach().abs().max().cpu().item()) if s_logits_f32.numel() else 0.0
                        t_abs = float(teacher_logits.detach().abs().max().cpu().item()) if teacher_logits.numel() else 0.0
                    print(
                        f" (dbg) epoch={epoch} batch={batch_idx+1}/{len(dataloader)} lr={lr_now} "
                        f"loss={loss_val:.6f} kd={kd_val:.6f} ce={ce_val:.6f} "
                        f"max|s_logit|={s_abs:.2f} max|t_logit|={t_abs:.2f} "
                        f"nonfinite_batches={nonfinite_batches}"
                    )

            print(f" KD Tradicional - poca {epoch}: Loss = {epoch_loss / max(1, len(dataloader)):.4f}")

            # Optional: save intermediate checkpoints (adapter + tokenizer) per epoch.
            if save_every_epoch and ckpt_root_path is not None:
                try:
                    out_dir = ckpt_root_path / f"epoch_{int(epoch) + 1:02d}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    student_model.save_pretrained(out_dir)
                    student_tokenizer.save_pretrained(out_dir)
                    # Write minimal metadata for traceability.
                    meta = {
                        "epoch": int(epoch) + 1,
                        "temperature": float(temperature),
                        "alpha": float(alpha),
                        "seed": int(seed),
                        "max_length": int(self.config.max_length),
                        "batch_size": int(batch_size),
                        "grad_accum_steps": int(grad_accum_steps),
                        "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                    }
                    (out_dir / "checkpoint_meta.json").write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                    print(f" (ckpt) Salvo checkpoint intermediário: {out_dir}")
                except Exception as exc:
                    print(f" (warn) Falha ao salvar checkpoint intermediário: {exc}")

        if debug_nan and (nonfinite_batches or nonfinite_teacher_shards or nonfinite_student_logits):
            print(
                " (dbg) Resumo numérico KD: "
                f"nonfinite_batches={nonfinite_batches}, "
                f"nonfinite_teacher_shards={nonfinite_teacher_shards}, "
                f"nonfinite_student_logits={nonfinite_student_logits}"
            )

        metrics["final_loss"] = metrics["losses"][-1] if metrics["losses"] else None
        return student_model, metrics


class ReasoningAwareDistiller:
    def __init__(self, config: EvidenceBasedConfig, cot_cache_path: Optional[str] = None, logits_cache_dir: Optional[str] = None):
        self.config = config
        self.cot_cache_path = cot_cache_path
        self.logits_cache_dir = logits_cache_dir

    def _load_cot_cache(self, path: str):
        import json

        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    class _LazyTeacherLogitsCache:
        def __init__(self, shard_paths, shard_sizes):
            self.shard_paths = list(shard_paths)
            self.shard_sizes = list(int(x) for x in shard_sizes)
            if len(self.shard_paths) != len(self.shard_sizes):
                raise ValueError("LazyTeacherLogitsCache: shard_paths e shard_sizes com tamanhos diferentes")

            # Prefix sums for global index -> shard mapping.
            self._starts = []
            total = 0
            for n in self.shard_sizes:
                self._starts.append(int(total))
                total += int(n)
            self._total = int(total)

            # Single-shard cache (keeps RAM bounded).
            self._cached_shard_idx = None
            self._cached_logits = None

        def __len__(self):
            return int(self._total)

        def _find_shard(self, idx: int) -> int:
            # Linear search is OK for small shard counts; for large counts, use bisect.
            import bisect

            i = bisect.bisect_right(self._starts, int(idx)) - 1
            if i < 0:
                i = 0
            return int(i)

        def shard_ranges(self):
            for shard_idx, (start, n) in enumerate(zip(self._starts, self.shard_sizes)):
                yield int(shard_idx), int(start), int(start + n)

        def _load_shard_logits(self, shard_idx: int) -> torch.Tensor:
            # Load only one shard at a time.
            payload = torch.load(self.shard_paths[int(shard_idx)], map_location="cpu")
            logits = payload.get("logits")
            if logits is None:
                raise ValueError(f"Shard sem 'logits': {self.shard_paths[int(shard_idx)]}")
            # Keep fp16 in RAM; convert on-the-fly in training.
            return logits

        def __getitem__(self, idx: int) -> torch.Tensor:
            if idx < 0:
                idx = int(self._total) + int(idx)
            if idx < 0 or idx >= int(self._total):
                raise IndexError(idx)

            shard_idx = self._find_shard(int(idx))
            if self._cached_shard_idx != int(shard_idx) or self._cached_logits is None:
                self._cached_logits = self._load_shard_logits(int(shard_idx))
                self._cached_shard_idx = int(shard_idx)

            local = int(idx) - int(self._starts[int(shard_idx)])
            return self._cached_logits[int(local)]

    def _estimate_logits_cache_size_mb(self, directory: str) -> float:
        from pathlib import Path

        stats_path = Path(directory) / "shard_stats.jsonl"
        if not stats_path.exists():
            return 0.0
        try:
            import json

            total_elems = 0
            for line in stats_path.read_text(encoding="utf-8").splitlines():
                line = (line or "").strip()
                if not line:
                    continue
                rec = json.loads(line)
                shape = rec.get("shape")
                if not isinstance(shape, list) or not shape:
                    continue
                elems = 1
                for d in shape:
                    try:
                        elems *= int(d)
                    except Exception:
                        elems = 0
                        break
                total_elems += int(elems)
            # Cache is fp16 on disk (2 bytes).
            return float(total_elems * 2) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _read_shard_sizes(self, directory: str, shard_paths):
        """Return per-shard batch sizes aligned to shard_paths.

        Prefers shard_stats.jsonl (fast). Falls back to loading each shard (slow).
        """

        from pathlib import Path

        # Parse shard index from filename.
        def _idx(p: Path) -> int:
            name = p.name
            try:
                # teacher_logits_shard_{i}.pt
                mid = name.split("teacher_logits_shard_")[-1]
                mid = mid.split(".pt")[0]
                return int(mid)
            except Exception:
                return -1

        stats_path = Path(directory) / "shard_stats.jsonl"
        if stats_path.exists():
            try:
                import json

                sizes_by_idx = {}
                for line in stats_path.read_text(encoding="utf-8").splitlines():
                    line = (line or "").strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    sidx = rec.get("shard")
                    shape = rec.get("shape")
                    if sidx is None or not isinstance(shape, list) or not shape:
                        continue
                    try:
                        sizes_by_idx[int(sidx)] = int(shape[0])
                    except Exception:
                        continue

                out = []
                for p in shard_paths:
                    out.append(int(sizes_by_idx.get(_idx(p), 0)))
                if all(int(x) > 0 for x in out):
                    return out
            except Exception:
                pass

        # Slow fallback: load each shard to read logits shape.
        out = []
        for p in shard_paths:
            payload = torch.load(p, map_location="cpu")
            logits = payload.get("logits")
            if logits is None:
                out.append(0)
            else:
                out.append(int(logits.size(0)))
        return out

    def _load_logits_cache(self, directory: str):
        """Load teacher logits cache.

        Modes:
        - eager: concatenates all shards into one tensor (legacy behavior, high RAM)
        - lazy: keeps shards on disk and loads one shard at a time (low RAM)
        - auto: chooses based on estimated cache size
        """

        from pathlib import Path

        mode = str(os.environ.get("SLM_LOGITS_CACHE_LOAD_MODE", "auto")).strip().lower()
        eager_max_mb = float(os.environ.get("SLM_LOGITS_CACHE_EAGER_MAX_MB", "2048"))

        shard_paths = sorted(Path(directory).glob("teacher_logits_shard_*.pt"))
        if not shard_paths:
            raise ValueError(f"Nenhum shard em {directory}")

        est_mb = self._estimate_logits_cache_size_mb(directory)
        if mode not in {"auto", "eager", "lazy"}:
            print(f"[WARN] SLM_LOGITS_CACHE_LOAD_MODE inválido: '{mode}'. Usando 'auto'.")
            mode = "auto"

        chosen = mode
        if mode == "auto":
            # If stats are missing, be conservative and stay eager for small shard counts.
            if est_mb > 0.0 and est_mb > eager_max_mb:
                chosen = "lazy"
            elif est_mb == 0.0 and len(shard_paths) > 16:
                chosen = "lazy"
            else:
                chosen = "eager"

        if chosen == "eager":
            tensors = []
            for shard in shard_paths:
                payload = torch.load(shard, map_location="cpu")
                tensors.append(payload["logits"].to(torch.float32))
            return torch.cat(tensors, dim=0)

        shard_sizes = self._read_shard_sizes(directory, shard_paths)
        total = sum(int(x) for x in shard_sizes)
        if total <= 0:
            raise ValueError(f"Não foi possível inferir tamanhos dos shards em {directory}")
        if est_mb > 0.0:
            print(f" (info) logits_cache load=lazy est_size_mb={est_mb:.1f} shards={len(shard_paths)}")
        else:
            print(f" (info) logits_cache load=lazy shards={len(shard_paths)}")
        return ReasoningAwareDistiller._LazyTeacherLogitsCache(shard_paths, shard_sizes)

    def _align_teacher_logits(self, teacher_logits: torch.Tensor, student_vocab: int) -> torch.Tensor:
        t_vocab = teacher_logits.size(-1)
        if t_vocab == student_vocab:
            return teacher_logits
        if t_vocab > student_vocab:
            return teacher_logits[..., :student_vocab]
        pad = teacher_logits.new_full((*teacher_logits.shape[:-1], student_vocab - t_vocab), -1e9)
        return torch.cat([teacher_logits, pad], dim=-1)

    def distill_with_reasoning(
        self,
        student_model,
        teacher_model,
        student_tokenizer,
        raw_dataset,
        seed: int = 42,
        use_cot_cache: bool = True,
        use_logits_cache: bool = True,
        use_teacher_logits: bool = True,
        granularity_level: int = 0,
        post_cot: bool = False,
        granularity_multi_level: bool = False,
        post_cot_use_ig: bool = False,
    ):
        set_seed(seed)
        device = self.config.device

        def _env_flag(name: str, default: str = "0") -> bool:
            v = os.environ.get(name, default)
            return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

        def _env_float(name: str, default: float) -> float:
            v = os.environ.get(name)
            if v is None:
                return float(default)
            try:
                return float(str(v).strip())
            except Exception:
                return float(default)

        def _env_int(name: str, default: int) -> int:
            v = os.environ.get(name)
            if v is None:
                return int(default)
            try:
                return int(str(v).strip())
            except Exception:
                return int(default)

        debug_nan = _env_flag("SLM_TRAIN_DEBUG_NAN", "1")
        log_every = max(1, _env_int("SLM_TRAIN_LOG_EVERY", 50))
        skip_nonfinite_batch = _env_flag("SLM_TRAIN_SKIP_NONFINITE_BATCH", "1")
        clip_grad_norm = float(self.config.kd_params.get("clip_grad_norm", 1.0))
        clip_grad_norm = float(os.environ.get("SLM_TRAIN_CLIP_GRAD_NORM", clip_grad_norm))
        max_logit_abs = _env_float("SLM_TRAIN_MAX_LOGIT_ABS", 100.0)
        apply_logit_sanitize = _env_flag("SLM_TRAIN_SANITIZE_LOGITS", "1")

        # Reasoning mask QA controls (backward-compatible by default).
        mask_fallback_to_completion = _env_flag("SLM_REASONING_MASK_FALLBACK_TO_COMPLETION", "1")
        mask_strict = _env_flag("SLM_REASONING_MASK_STRICT", "0")
        mask_max_fallback = _env_float("SLM_REASONING_MASK_MAX_FALLBACK", 0.30)
        mask_min_reasoning_frac = _env_float("SLM_REASONING_MASK_MIN_REASONING_FRAC", 0.02)

        def _sanitize_logits(x: torch.Tensor) -> torch.Tensor:
            if not apply_logit_sanitize:
                return x
            x = torch.nan_to_num(x, nan=0.0, posinf=max_logit_abs, neginf=-max_logit_abs)
            if max_logit_abs > 0:
                x = x.clamp(min=-max_logit_abs, max=max_logit_abs)
            return x

        def _has_nonfinite(x: torch.Tensor) -> bool:
            try:
                return bool((~torch.isfinite(x)).any().item())
            except Exception:
                return True

        logits_ready = bool(use_teacher_logits and use_logits_cache and self.logits_cache_dir and os.path.isdir(self.logits_cache_dir))
        cot_ready = bool(use_cot_cache and self.cot_cache_path and os.path.exists(self.cot_cache_path))
        if use_cot_cache and not cot_ready:
            raise ValueError("cot_cache_path no encontrado. Gere o cache primeiro.")
        if use_teacher_logits and not logits_ready and teacher_model is None:
            raise ValueError("Precisa de teacher_model quando não há cache de logits e use_teacher_logits=True.")

        teacher_logits_cache = self._load_logits_cache(self.logits_cache_dir) if logits_ready else None
        cot_records = self._load_cot_cache(self.cot_cache_path) if cot_ready else None

        # CoT/logits cache QA: if examples built from CoT don't align with cached logits,
        # logits-KD becomes scientifically invalid (silent misalignment). Default is to
        # warn and disable cache; strict mode aborts.
        cot_strict = _env_flag("SLM_COT_CACHE_STRICT", "0")
        require_exact_logits_match = _env_flag("SLM_LOGITS_CACHE_REQUIRE_EXACT_MATCH", "0")

        # Build prompt + target sequences
        examples: List[Dict[str, str]] = []
        if cot_records is not None:
            # Basic cache integrity audit (does NOT filter records; filtering would desync logits cache).
            missing_prompt = 0
            missing_reasoning = 0
            missing_answer = 0
            bad_json = 0
            for rec in cot_records:
                if not isinstance(rec, dict):
                    bad_json += 1
                    continue
                p = rec.get("prompt")
                r = rec.get("teacher_reasoning")
                a = rec.get("teacher_answer")
                if not (isinstance(p, str) and p.strip()) and not isinstance(rec.get("prompt_levels"), dict):
                    missing_prompt += 1
                if not (isinstance(r, str) and r.strip()) and not isinstance(rec.get("teacher_reasoning_levels"), dict):
                    missing_reasoning += 1
                if not (isinstance(a, str) and a.strip()) and not isinstance(rec.get("teacher_answer_levels"), dict):
                    missing_answer += 1

            if bad_json:
                msg = f"CoT cache contém entradas não-dict: {bad_json}/{len(cot_records)}"
                if bool(cot_strict):
                    raise ValueError(msg)
                print(f"[WARN] {msg}")
            if missing_prompt or missing_reasoning or missing_answer:
                msg = (
                    "CoT cache parece incompleto (pode causar skips e desalinhamento de logits). "
                    f"missing_prompt={missing_prompt}, missing_reasoning={missing_reasoning}, missing_answer={missing_answer}, total={len(cot_records)}"
                )
                if bool(cot_strict):
                    raise ValueError(msg)
                print(f"[WARN] {msg}")

            for rec in cot_records:
                prompt_levels = rec.get("prompt_levels")
                reasoning_levels = rec.get("teacher_reasoning_levels")
                answer_levels = rec.get("teacher_answer_levels")

                if bool(granularity_multi_level) and isinstance(prompt_levels, dict) and isinstance(reasoning_levels, dict):
                    levels = rec.get("granularity_levels")
                    if not isinstance(levels, list) or not levels:
                        try:
                            levels = sorted({int(k) for k in prompt_levels.keys()})
                        except Exception:
                            levels = []

                    for lvl in levels:
                        k = str(int(lvl))
                        prompt = str(prompt_levels.get(k) or "").strip()
                        if not prompt:
                            continue
                        reasoning = str(reasoning_levels.get(k) or "").strip()
                        if not reasoning:
                            continue
                        answer = str((answer_levels or {}).get(k) or rec.get("teacher_answer", "") or "").strip()

                        if bool(post_cot):
                            reasoning_use = reasoning
                            if bool(post_cot_use_ig) and isinstance(rec.get("teacher_reasoning_ig"), str) and rec.get("teacher_reasoning_ig").strip():
                                reasoning_use = str(rec.get("teacher_reasoning_ig") or "").strip()
                            teacher_full = (answer + "\n### REASONING:\n" + reasoning_use)
                        else:
                            teacher_full = (reasoning + "\n### FINAL_ANSWER: " + answer)
                        examples.append({"prompt": prompt, "teacher_full": teacher_full})
                    continue

                prompt = rec.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    prompt, _ = build_cot_prompt(
                        str(rec.get("question", rec.get("text", ""))),
                        granularity_level=int(granularity_level or 0),
                        post_cot=bool(post_cot),
                    )
                if bool(post_cot):
                    reasoning = rec.get("teacher_reasoning", "").strip()
                    if bool(post_cot_use_ig) and isinstance(rec.get("teacher_reasoning_ig"), str) and rec.get("teacher_reasoning_ig").strip():
                        reasoning = str(rec.get("teacher_reasoning_ig") or "").strip()
                    teacher_full = (rec.get("teacher_answer", "").strip() + "\n### REASONING:\n" + reasoning)
                else:
                    teacher_full = (rec.get("teacher_reasoning", "").strip() + "\n### FINAL_ANSWER: " + rec.get("teacher_answer", ""))
                examples.append({"prompt": prompt, "teacher_full": teacher_full})
        else:
            for ex in raw_dataset:
                prompt, _ = build_cot_prompt(
                    str(ex.get("question", ex.get("text", ""))),
                    granularity_level=int(granularity_level or 0),
                    post_cot=bool(post_cot),
                )
                examples.append({"prompt": prompt, "teacher_full": str(ex.get("answer", ""))})

        prompts = [e["prompt"] for e in examples]
        targets = [e["teacher_full"] for e in examples]
        full_sequences = [p + t for p, t in zip(prompts, targets)]

        tok_prompts = student_tokenizer(prompts, truncation=True, padding="max_length", max_length=self.config.max_length, return_tensors="pt")
        prompt_lengths = tok_prompts["attention_mask"].sum(dim=1)

        tok_full = student_tokenizer(full_sequences, truncation=True, padding="max_length", max_length=self.config.max_length, return_tensors="pt")
        labels = tok_full["input_ids"].clone()
        labels[tok_full["attention_mask"] == 0] = -100
        positions = torch.arange(labels.size(1)).unsqueeze(0)
        prompt_lengths = torch.clamp(prompt_lengths, max=labels.size(1)).unsqueeze(1)
        labels[positions < prompt_lengths] = -100

        # KD mask should apply only to the reasoning portion (not the answer).
        # CE still trains the entire completion; this keeps backward-compat while
        # making KD interpretation cleaner.
        reasoning_mask = torch.zeros_like(labels, dtype=torch.long)

        # Audit stats (helps detect truncation/format drift silently breaking the mask).
        mask_audit = {
            "n_examples": int(len(full_sequences)),
            "n_fallback": 0,
            "n_mask_empty": 0,
            "n_reasoning_marker_found": 0,
            "n_final_marker_found": 0,
            "reasoning_tokens_sum": 0,
            "completion_tokens_sum": 0,
        }

        def _safe_find(hay: str, needle: str) -> int:
            try:
                return (hay or "").lower().find((needle or "").lower())
            except Exception:
                return -1

        for i, seq in enumerate(full_sequences):
            # Default fallback: if we can't locate markers, KD over completion.
            fallback_mask = (labels[i] != -100).long()
            mask_audit["completion_tokens_sum"] += int(fallback_mask.sum().item())

            if bool(post_cot):
                # Format: prompt + "### FINAL_ANSWER:\n" + answer + "\n### REASONING:\n" + reasoning
                m = "### REASONING:"
                start = _safe_find(seq, m)
                if start < 0:
                    if mask_fallback_to_completion:
                        reasoning_mask[i] = fallback_mask
                        mask_audit["n_fallback"] += 1
                        mask_audit["reasoning_tokens_sum"] += int(fallback_mask.sum().item())
                    else:
                        mask_audit["n_mask_empty"] += 1
                    continue

                mask_audit["n_reasoning_marker_found"] += 1

                prefix = seq[: start + len(m)]
                tok_prefix = student_tokenizer(prefix, truncation=True, padding="max_length", max_length=self.config.max_length, return_tensors="pt")
                start_len = int(tok_prefix["attention_mask"].sum().item())

                positions_i = torch.arange(labels.size(1))
                mask_i = (positions_i >= start_len) & (labels[i] != -100)
                reasoning_mask[i][mask_i] = 1
                rs = int(reasoning_mask[i].sum().item())
                if rs <= 0:
                    # Likely truncation: marker exists in raw text, but its token-space span fell outside max_length.
                    mask_audit["n_mask_empty"] += 1
                    if mask_fallback_to_completion:
                        reasoning_mask[i] = fallback_mask
                        mask_audit["n_fallback"] += 1
                        rs = int(fallback_mask.sum().item())
                mask_audit["reasoning_tokens_sum"] += int(rs)
                continue

            # Non-post: prompt + "### REASONING:\n" + reasoning + "\n### FINAL_ANSWER: " + answer
            m_start = "### REASONING:"
            m_end = "### FINAL_ANSWER:"
            start = _safe_find(seq, m_start)
            end = _safe_find(seq, m_end)
            if start < 0:
                if mask_fallback_to_completion:
                    reasoning_mask[i] = fallback_mask
                    mask_audit["n_fallback"] += 1
                    mask_audit["reasoning_tokens_sum"] += int(fallback_mask.sum().item())
                else:
                    mask_audit["n_mask_empty"] += 1
                continue

            mask_audit["n_reasoning_marker_found"] += 1

            prefix = seq[: start + len(m_start)]
            tok_prefix = student_tokenizer(prefix, truncation=True, padding="max_length", max_length=self.config.max_length, return_tensors="pt")
            start_len = int(tok_prefix["attention_mask"].sum().item())

            end_len = labels.size(1)
            if end > start:
                mask_audit["n_final_marker_found"] += 1
                prefix_end = seq[:end]
                tok_end = student_tokenizer(prefix_end, truncation=True, padding="max_length", max_length=self.config.max_length, return_tensors="pt")
                end_len = int(tok_end["attention_mask"].sum().item())
                end_len = max(start_len, min(end_len, labels.size(1)))

            positions_i = torch.arange(labels.size(1))
            mask_i = (positions_i >= start_len) & (positions_i < end_len) & (labels[i] != -100)
            reasoning_mask[i][mask_i] = 1
            rs = int(reasoning_mask[i].sum().item())
            if rs <= 0:
                mask_audit["n_mask_empty"] += 1
                if mask_fallback_to_completion:
                    reasoning_mask[i] = fallback_mask
                    mask_audit["n_fallback"] += 1
                    rs = int(fallback_mask.sum().item())
            mask_audit["reasoning_tokens_sum"] += int(rs)

        class COTDataset(TorchDataset):
            def __init__(self, inputs, labels, reasoning_mask, teacher_logits):
                self.inputs = inputs
                self.labels = labels
                self.reasoning_mask = reasoning_mask
                self.teacher_logits = teacher_logits

            def __len__(self):
                return self.inputs["input_ids"].size(0)

            def __getitem__(self, idx):
                item = {
                    "input_ids": self.inputs["input_ids"][idx],
                    "attention_mask": self.inputs["attention_mask"][idx],
                    "labels": self.labels[idx],
                    "reasoning_mask": self.reasoning_mask[idx],
                }
                if self.teacher_logits is not None:
                    item["teacher_logits"] = self.teacher_logits[idx]
                return item

        if teacher_logits_cache is not None:
            try:
                cached_n = int(len(teacher_logits_cache))
            except Exception:
                try:
                    cached_n = int(teacher_logits_cache.size(0))
                except Exception:
                    cached_n = -1

            built_n = int(len(examples))
            if cached_n >= 0 and cached_n != built_n:
                msg = (
                    "Logits cache não alinha com exemplos construídos do CoT. "
                    f"cached_n={cached_n} built_n={built_n}. "
                    "Isso invalidaria logits-KD (teacher logits em exemplos errados)."
                )
                if bool(require_exact_logits_match):
                    raise ValueError(msg + " Defina SLM_LOGITS_CACHE_REQUIRE_EXACT_MATCH=0 para desabilitar o cache automaticamente.")
                print(f"[WARN] {msg} Desabilitando logits_cache e usando teacher_model on-the-fly.")
                teacher_logits_cache = None

        # Use shard-aware deterministic shuffling when we have a lazy cache.
        sampler = None
        is_lazy_cache = False
        try:
            is_lazy_cache = isinstance(teacher_logits_cache, ReasoningAwareDistiller._LazyTeacherLogitsCache)
        except Exception:
            is_lazy_cache = False

        if bool(is_lazy_cache):
            allow_shuffle = _env_flag("SLM_LOGITS_CACHE_ALLOW_SHUFFLE", "0")
            if not allow_shuffle:
                # Deterministic shard-wise shuffle: shuffle shards and indices within shards
                # to keep IO bounded (avoid random shard thrashing).
                class _ShardShuffleSampler(torch.utils.data.Sampler):
                    def __init__(self, lazy_cache, *, base_seed: int):
                        self.lazy_cache = lazy_cache
                        self.base_seed = int(base_seed)
                        self.epoch = 0

                    def set_epoch(self, epoch: int) -> None:
                        self.epoch = int(epoch)

                    def __iter__(self):
                        import random

                        rng = random.Random(int(self.base_seed) + int(self.epoch))
                        shards = [(s, a, b) for (s, a, b) in self.lazy_cache.shard_ranges()]
                        rng.shuffle(shards)
                        for (_sid, start, end) in shards:
                            idxs = list(range(int(start), int(end)))
                            rng.shuffle(idxs)
                            for i in idxs:
                                yield int(i)

                    def __len__(self):
                        return int(len(self.lazy_cache))

                sampler = _ShardShuffleSampler(teacher_logits_cache, base_seed=int(seed))
            else:
                print("[WARN] SLM_LOGITS_CACHE_ALLOW_SHUFFLE=1 com cache lazy pode causar IO alto.")

        dataset = COTDataset(tok_full, labels, reasoning_mask, teacher_logits_cache)

        tok_vocab = len(student_tokenizer)
        model_vocab = student_model.get_input_embeddings().weight.size(0)
        # Only resize when tokenizer requires *more* tokens than the model provides.
        # Many HF models reserve extra logits rows; shrinking can be unsupported and
        # can desync output vocab size vs embeddings.
        if tok_vocab > model_vocab:
            print(f" Resizing student embeddings (expand): tokenizer={tok_vocab}, model={model_vocab}")
            student_model.resize_token_embeddings(tok_vocab)

        # Training stability + memory: disable KV cache and enable gradient checkpointing.
        try:
            student_model.config.use_cache = False
        except Exception:
            pass
        try:
            student_model.gradient_checkpointing_enable()
        except Exception:
            pass

        # If the base model is 4-bit/8-bit, use PEFT k-bit preparation (QLoRA style).
        if getattr(student_model, "is_loaded_in_4bit", False) or getattr(student_model, "is_loaded_in_8bit", False):
            from peft import prepare_model_for_kbit_training

            student_model = prepare_model_for_kbit_training(student_model)

        lora_cfg = LoraConfig(
            r=self.config.kd_params["lora_rank"],
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        student_model = get_peft_model(student_model, lora_cfg)
        student_model.train()
        student_model = safe_model_to(student_model, device)
        if teacher_model is not None:
            teacher_model.eval()
            teacher_model = safe_model_to(teacher_model, device)
            try:
                teacher_model.config.use_cache = False
            except Exception:
                pass

        batch_size = int(self.config.kd_params.get("batch_size", 2))
        grad_accum_steps = int(self.config.kd_params.get("grad_accum_steps", 1))
        num_workers = int(self.config.kd_params.get("dataloader_num_workers", 0))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers,
        )

        num_epochs = self.config.kd_params["epochs"]
        num_training_steps = num_epochs * len(dataloader)
        warmup_steps = int(0.1 * num_training_steps)

        optimizer = torch.optim.AdamW(student_model.parameters(), lr=self.config.kd_params["learning_rates"]["kd"])
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        scaler = make_grad_scaler(device)

        temperature_schedule = self.config.kd_params.get("temperature_schedule") or [1.0]
        alpha_schedule = self.config.kd_params.get("alpha_schedule") or [0.5]

        # Attach audit stats early so they get returned even if training aborts.
        def _safe_div(num: float, den: float) -> float:
            return float(num) / float(den) if float(den) != 0.0 else 0.0

        mask_audit["reasoning_tokens_mean"] = _safe_div(mask_audit["reasoning_tokens_sum"], mask_audit["n_examples"])
        mask_audit["completion_tokens_mean"] = _safe_div(mask_audit["completion_tokens_sum"], mask_audit["n_examples"])
        mask_audit["reasoning_token_fraction_mean"] = _safe_div(mask_audit["reasoning_tokens_sum"], mask_audit["completion_tokens_sum"])
        mask_audit["fallback_rate"] = _safe_div(mask_audit["n_fallback"], mask_audit["n_examples"])
        mask_audit["reasoning_marker_found_rate"] = _safe_div(mask_audit["n_reasoning_marker_found"], mask_audit["n_examples"])
        # For non-post mode, this checks whether FINAL marker was present; for post mode it may stay 0.
        mask_audit["final_marker_found_rate"] = _safe_div(mask_audit["n_final_marker_found"], mask_audit["n_examples"])

        # Optional enforcement: avoid silently running a "reasoning" experiment with a broken mask.
        if mask_audit["n_examples"] > 0:
            fallback_rate = float(mask_audit.get("fallback_rate") or 0.0)
            reasoning_frac = float(mask_audit.get("reasoning_token_fraction_mean") or 0.0)
            if (fallback_rate > float(mask_max_fallback)) or (reasoning_frac < float(mask_min_reasoning_frac)):
                msg = (
                    "Reasoning mask quality appears low. "
                    f"fallback_rate={fallback_rate:.3f} (max={float(mask_max_fallback):.3f}), "
                    f"reasoning_frac={reasoning_frac:.3f} (min={float(mask_min_reasoning_frac):.3f}), "
                    f"n_mask_empty={int(mask_audit.get('n_mask_empty') or 0)}. "
                    "This can happen due to truncation (max_length) or format drift in cached CoT."
                )
                if bool(mask_strict):
                    raise ValueError(
                        msg
                        + " Set SLM_REASONING_MASK_STRICT=0 to allow running, or adjust: "
                        + "SLM_REASONING_MASK_MAX_FALLBACK / SLM_REASONING_MASK_MIN_REASONING_FRAC / max_length."
                    )
                print(f"[WARN] {msg}")

        metrics = {
            "losses": [],
            "kd_losses": [],
            "ce_losses": [],
            "use_teacher_logits": bool(use_teacher_logits),
            "reasoning_mask_audit": mask_audit,
            "numeric_stability": {
                "train_sanitize_logits": bool(apply_logit_sanitize),
                "train_max_logit_abs": float(max_logit_abs),
            },
            "reasoning_mask_controls": {
                "fallback_to_completion": bool(mask_fallback_to_completion),
                "strict": bool(mask_strict),
                "max_fallback": float(mask_max_fallback),
                "min_reasoning_frac": float(mask_min_reasoning_frac),
            },
            "logits_cache_controls": {
                "logits_ready": bool(logits_ready),
                "cot_ready": bool(cot_ready),
                "load_mode": str(os.environ.get("SLM_LOGITS_CACHE_LOAD_MODE", "auto")),
                "eager_max_mb": float(os.environ.get("SLM_LOGITS_CACHE_EAGER_MAX_MB", "2048")),
                "lazy": bool(is_lazy_cache),
                "cot_cache_strict": bool(cot_strict),
                "require_exact_logits_match": bool(require_exact_logits_match),
                "allow_shuffle": bool(_env_flag("SLM_LOGITS_CACHE_ALLOW_SHUFFLE", "0")),
            },
        }

        nonfinite_batches = 0
        nonfinite_teacher = 0
        nonfinite_student = 0

        for epoch in range(num_epochs):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                try:
                    sampler.set_epoch(int(epoch))
                except Exception:
                    pass
            temperature = get_schedule_value(temperature_schedule, epoch, default=1.0)
            alpha = get_schedule_value(alpha_schedule, epoch, default=0.5)

            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"CoT KD Epoch {epoch}")):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                labels = batch["labels"].to(device)
                reasoning_mask = batch["reasoning_mask"].to(device)

                # Forward can be AMP, but compute losses in fp32 outside autocast.
                with autocast_ctx(device):
                    s_logits = student_model(**inputs).logits

                s_logits_f32 = _sanitize_logits(s_logits.to(dtype=torch.float32))
                if _has_nonfinite(s_logits_f32):
                    nonfinite_student += 1
                    if debug_nan:
                        print(f" (warn) student logits não-finitos (reasoning) epoch={epoch} batch={batch_idx}")
                    s_logits_f32 = _sanitize_logits(s_logits_f32)

                vocab_size = s_logits_f32.size(-1)

                labels_ce = labels.long().clamp(min=-100, max=vocab_size - 1)
                labels_ce, _san = sanitize_labels_for_ce(labels_ce, vocab_size)

                ce_loss = F.cross_entropy(s_logits_f32.reshape(-1, vocab_size), labels_ce.reshape(-1), ignore_index=-100)

                if use_teacher_logits:
                    if "teacher_logits" in batch:
                        teacher_logits = batch["teacher_logits"].to(device=device, dtype=torch.float32)
                    else:
                        with torch.inference_mode():
                            teacher_logits = teacher_model(**inputs).logits
                            teacher_logits = teacher_logits.to(dtype=torch.float32)
                    teacher_logits = self._align_teacher_logits(teacher_logits, vocab_size)
                    if _has_nonfinite(teacher_logits):
                        nonfinite_teacher += 1
                        if debug_nan:
                            print(f" (warn) teacher logits não-finitos (reasoning) epoch={epoch} batch={batch_idx}")
                    teacher_logits = _sanitize_logits(teacher_logits)

                    t_probs = F.softmax(teacher_logits / float(temperature), dim=-1)
                    s_logp = F.log_softmax(s_logits_f32 / float(temperature), dim=-1)
                    token_kl = F.kl_div(s_logp, t_probs, reduction="none")
                    mask = reasoning_mask.unsqueeze(-1).float()
                    kd_loss = (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
                    kd_loss *= temperature**2
                    loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
                else:
                    kd_loss = torch.tensor(0.0, device=device)
                    loss = ce_loss

                if _has_nonfinite(loss) or _has_nonfinite(ce_loss) or _has_nonfinite(kd_loss):
                    nonfinite_batches += 1
                    if debug_nan:
                        lr_now = None
                        try:
                            lr_now = float(lr_scheduler.get_last_lr()[0])
                        except Exception:
                            pass
                        print(
                            " (warn) loss não-finita (reasoning); pulando batch. "
                            f"epoch={epoch} batch={batch_idx} lr={lr_now} "
                            f"ce={'nonfinite' if _has_nonfinite(ce_loss) else float(ce_loss.detach().cpu().item())} "
                            f"kd={'nonfinite' if _has_nonfinite(kd_loss) else float(kd_loss.detach().cpu().item())}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if not skip_nonfinite_batch:
                        raise RuntimeError("Encontrado loss/logits não-finitos durante CoT/reasoning KD.")
                    continue

                if grad_accum_steps > 1:
                    loss = loss / float(grad_accum_steps)

                scaler.scale(loss).backward()

                do_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
                if do_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), float(clip_grad_norm))
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm_val = float(grad_norm.detach().cpu().item())
                    else:
                        grad_norm_val = float(grad_norm)
                    if (not (grad_norm_val == grad_norm_val)) or (grad_norm_val == float("inf")):
                        nonfinite_batches += 1
                        if debug_nan:
                            print(f" (warn) grad_norm não-finito (reasoning); pulando step. epoch={epoch} batch={batch_idx}")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                try:
                    metrics["losses"].append(float((loss.detach().cpu().item() * grad_accum_steps) if grad_accum_steps > 1 else loss.detach().cpu().item()))
                    metrics["kd_losses"].append(float(kd_loss.detach().cpu().item()))
                    metrics["ce_losses"].append(float(ce_loss.detach().cpu().item()))
                except Exception:
                    pass

                if debug_nan and ((batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(dataloader)):
                    with torch.no_grad():
                        s_abs = float(s_logits_f32.detach().abs().max().cpu().item()) if s_logits_f32.numel() else 0.0
                        t_abs = float(teacher_logits.detach().abs().max().cpu().item()) if (use_teacher_logits and "teacher_logits" in locals() and teacher_logits.numel()) else 0.0
                    print(
                        f" (dbg) reasoning epoch={epoch} batch={batch_idx+1}/{len(dataloader)} "
                        f"loss={metrics['losses'][-1] if metrics['losses'] else float('nan'):.6f} "
                        f"kd={metrics['kd_losses'][-1] if metrics['kd_losses'] else float('nan'):.6f} "
                        f"ce={metrics['ce_losses'][-1] if metrics['ce_losses'] else float('nan'):.6f} "
                        f"max|s_logit|={s_abs:.2f} max|t_logit|={t_abs:.2f} nonfinite_batches={nonfinite_batches}"
                    )

            print(f" CoT KD - poca {epoch} concluda")

        if debug_nan and (nonfinite_batches or nonfinite_teacher or nonfinite_student):
            print(
                " (dbg) Resumo numérico CoT/reasoning: "
                f"nonfinite_batches={nonfinite_batches}, nonfinite_teacher={nonfinite_teacher}, nonfinite_student={nonfinite_student}"
            )

        metrics["final_loss"] = metrics["losses"][-1] if metrics["losses"] else None
        return student_model, metrics
