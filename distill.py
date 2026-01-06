from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import get_scheduler

from config import (
    EvidenceBasedConfig,
    get_schedule_value,
    safe_model_to,
    set_seed,
)


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


def build_reasoning_full_sequences_from_cot(cot_path: str, max_records: Optional[int] = None) -> List[str]:
    """Build prompt+completion sequences from a teacher CoT cache.

    Scientific validity fix: reasoning-mode logits cache (if enabled) must be
    computed on the same input sequences used for training.
    """

    import json

    seqs: List[str] = []
    with open(cot_path, "r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            prompt = (
                f"Q: {rec.get('question', rec.get('text', ''))}\n"
                "A: Let's think step by step.\n### REASONING:\n"
            )
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
        if tok_vocab != model_vocab:
            print(f" Resizing student embeddings: tokenizer={tok_vocab}, model={model_vocab}")
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
        # AMP: keep existing fp16 scaler behavior for compatibility.
        scaler = GradScaler()

        temperature_schedule = self.config.kd_params.get("temperature_schedule") or [3.0]
        alpha_schedule = self.config.kd_params.get("alpha_schedule") or [0.7]

        metrics = {"losses": [], "kd_losses": [], "ce_losses": []}

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
                    teacher_logits = data["logits"].to(device)
                    teacher_logits = self._align_teacher_logits(teacher_logits, vocab_size)
                else:
                    with torch.no_grad():
                        t_out = teacher_model(**inputs)
                        teacher_logits = self._align_teacher_logits(t_out.logits, vocab_size)

                with autocast():
                    s_logits = student_model(**inputs).logits
                    vocab_size = s_logits.size(-1)
                    ce_loss = F.cross_entropy(s_logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)
                    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
                    s_logp = F.log_softmax(s_logits / temperature, dim=-1)
                    token_kl = F.kl_div(s_logp, t_probs, reduction="none")
                    mask = (labels != -100).unsqueeze(-1).float()
                    kd_loss = (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
                    kd_loss *= temperature**2
                    loss = alpha * kd_loss + (1.0 - alpha) * ce_loss

                    if grad_accum_steps > 1:
                        loss = loss / float(grad_accum_steps)

                scaler.scale(loss).backward()

                do_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
                if do_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                metrics["losses"].append(float(loss.item()))
                metrics["kd_losses"].append(float(kd_loss.item()))
                metrics["ce_losses"].append(float(ce_loss.item()))
                # Track unscaled loss for reporting.
                epoch_loss += float((loss.item() * grad_accum_steps) if grad_accum_steps > 1 else loss.item())

            print(f" KD Tradicional - poca {epoch}: Loss = {epoch_loss / max(1, len(dataloader)):.4f}")

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

    def _load_logits_cache(self, directory: str) -> torch.Tensor:
        from pathlib import Path

        shard_paths = sorted(Path(directory).glob("teacher_logits_shard_*.pt"))
        if not shard_paths:
            raise ValueError(f"Nenhum shard em {directory}")
        tensors = []
        for shard in shard_paths:
            payload = torch.load(shard, map_location="cpu")
            tensors.append(payload["logits"].to(torch.float32))
        return torch.cat(tensors, dim=0)

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
    ):
        set_seed(seed)
        device = self.config.device

        logits_ready = bool(use_logits_cache and self.logits_cache_dir and os.path.isdir(self.logits_cache_dir))
        cot_ready = bool(use_cot_cache and self.cot_cache_path and os.path.exists(self.cot_cache_path))
        if use_cot_cache and not cot_ready:
            raise ValueError("cot_cache_path no encontrado. Gere o cache primeiro.")
        if not logits_ready and teacher_model is None:
            raise ValueError("Precisa de teacher_model quando no h cache de logits.")

        teacher_logits_cache = self._load_logits_cache(self.logits_cache_dir) if logits_ready else None
        cot_records = self._load_cot_cache(self.cot_cache_path) if cot_ready else None

        # Build prompt + target sequences
        examples: List[Dict[str, str]] = []
        if cot_records is not None:
            for rec in cot_records:
                prompt = (
                    f"Q: {rec.get('question', rec.get('text', ''))}\n"
                    "A: Let's think step by step.\n### REASONING:\n"
                )
                teacher_full = (rec.get("teacher_reasoning", "").strip() + "\n### FINAL_ANSWER: " + rec.get("teacher_answer", ""))
                examples.append({"prompt": prompt, "teacher_full": teacher_full})
        else:
            for ex in raw_dataset:
                prompt = (
                    f"Q: {ex.get('question', ex.get('text', ''))}\n"
                    "A: Let's think step by step.\n### REASONING:\n"
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
        reasoning_mask = (labels != -100).long()

        class COTDataset(TorchDataset):
            def __init__(self, inputs, labels, reasoning_mask, teacher_logits: Optional[torch.Tensor]):
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
            teacher_logits_cache = teacher_logits_cache[: len(examples)]

        dataset = COTDataset(tok_full, labels, reasoning_mask, teacher_logits_cache)

        tok_vocab = len(student_tokenizer)
        model_vocab = student_model.get_input_embeddings().weight.size(0)
        if tok_vocab != model_vocab:
            print(f" Resizing student embeddings: tokenizer={tok_vocab}, model={model_vocab}")
            student_model.resize_token_embeddings(tok_vocab)

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

        dataloader = DataLoader(dataset, batch_size=self.config.kd_params["batch_size"], shuffle=True, pin_memory=True, num_workers=2)

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
        scaler = GradScaler()

        temperature_schedule = self.config.kd_params.get("temperature_schedule") or [1.0]
        alpha_schedule = self.config.kd_params.get("alpha_schedule") or [0.5]

        metrics = {"losses": [], "kd_losses": [], "ce_losses": []}

        for epoch in range(num_epochs):
            temperature = get_schedule_value(temperature_schedule, epoch, default=1.0)
            alpha = get_schedule_value(alpha_schedule, epoch, default=0.5)

            for batch in tqdm(dataloader, desc=f"CoT KD Epoch {epoch}"):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                labels = batch["labels"].to(device)
                reasoning_mask = batch["reasoning_mask"].to(device)

                vocab_size = student_model.get_input_embeddings().weight.size(0)
                labels = labels.long().clamp(min=-100, max=vocab_size - 1)
                labels, _san = sanitize_labels_for_ce(labels, vocab_size)

                if "teacher_logits" in batch:
                    teacher_logits = self._align_teacher_logits(batch["teacher_logits"].to(device), vocab_size)
                else:
                    with torch.inference_mode():
                        teacher_logits = teacher_model(**inputs).logits

                with autocast():
                    s_logits = student_model(**inputs).logits
                    vocab_size = s_logits.size(-1)
                    ce_loss = F.cross_entropy(s_logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)
                    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
                    s_logp = F.log_softmax(s_logits / temperature, dim=-1)
                    token_kl = F.kl_div(s_logp, t_probs, reduction="none")
                    mask = reasoning_mask.unsqueeze(-1).float()
                    kd_loss = (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
                    kd_loss *= temperature**2
                    loss = alpha * kd_loss + (1.0 - alpha) * ce_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                metrics["losses"].append(float(loss.item()))
                metrics["kd_losses"].append(float(kd_loss.item()))
                metrics["ce_losses"].append(float(ce_loss.item()))

            print(f" CoT KD - poca {epoch} concluda")

        metrics["final_loss"] = metrics["losses"][-1] if metrics["losses"] else None
        return student_model, metrics
