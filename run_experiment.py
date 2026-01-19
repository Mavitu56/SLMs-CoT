from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cache import cache_teacher_cot, cache_teacher_logits, read_cache_metadata, tokenizer_fingerprint
from config import EvidenceBasedConfig, GenerationConfig, ensure_tokenizer_has_pad, get_safe_tokenizer_length, safe_model_to, set_seed
from data import load_training_dataset
from distill import ReasoningAwareDistiller, TraditionalKDDistiller, autocast_ctx, make_grad_scaler, preprocess_and_tokenize
from prompts import build_cascod_answer_prompt, build_cascod_rationale_prompt
from eval import StandardizedEvaluator
from report import ScientificLogger, write_plots, write_report_json, write_summary_txt
from stats import StatisticalAnalyst


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _collect_environment_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    # Optional libs (best-effort).
    try:
        import torch as _torch

        meta["torch_version"] = getattr(_torch, "__version__", None)
        meta["cuda_available"] = bool(_torch.cuda.is_available())
        meta["cuda_version"] = getattr(_torch.version, "cuda", None)
        try:
            meta["cudnn_version"] = int(_torch.backends.cudnn.version() or 0)
        except Exception:
            meta["cudnn_version"] = None
        if bool(_torch.cuda.is_available()):
            try:
                meta["gpu_name"] = str(_torch.cuda.get_device_name(0))
            except Exception:
                meta["gpu_name"] = None
    except Exception:
        pass

    try:
        import transformers as _transformers

        meta["transformers_version"] = getattr(_transformers, "__version__", None)
    except Exception:
        pass
    try:
        import peft as _peft

        meta["peft_version"] = getattr(_peft, "__version__", None)
    except Exception:
        pass
    try:
        import datasets as _datasets

        meta["datasets_version"] = getattr(_datasets, "__version__", None)
    except Exception:
        pass
    return meta


def _train_sft_lora(
    cfg: EvidenceBasedConfig,
    model,
    tokenizer,
    raw_dataset,
    *,
    seed: int,
) -> tuple[Any, Dict[str, Any]]:
    """Supervised fine-tuning (SFT) helper.

    Baseline 0.1 requirement: explicit teacher fine-tuning on TRAIN split only,
    with clear checkpoint saving and reproducibility.

    Implementation choice: LoRA/QLoRA-style adapters for Colab feasibility.
    """

    import math

    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import get_scheduler

    set_seed(seed)
    device = cfg.device

    tokenized = preprocess_and_tokenize(raw_dataset, tokenizer, max_length=cfg.max_length)

    # Stability + memory.
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # QLoRA preparation when model is k-bit.
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        try:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        except Exception:
            # If unavailable, continue; training may still work but will be less stable.
            pass

    if not hasattr(model, "peft_config"):
        lora_cfg = LoraConfig(
            r=int(cfg.kd_params.get("lora_rank", 16)),
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
    model.train()
    model = safe_model_to(model, device)

    batch_size = int(cfg.kd_params.get("batch_size", 2))
    grad_accum_steps = int(cfg.kd_params.get("grad_accum_steps", 1))
    num_workers = int(cfg.kd_params.get("dataloader_num_workers", 0))
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    num_epochs = int(cfg.kd_params.get("epochs", 1))
    num_training_steps = max(1, num_epochs * len(dataloader))
    warmup_steps = max(0, int(0.1 * num_training_steps))

    lr = float(cfg.kd_params.get("learning_rates", {}).get("kd", 5e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = make_grad_scaler(device)

    def _env_flag(name: str, default: str = "0") -> bool:
        v = os.environ.get(name, default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        if v is None:
            return int(default)
        try:
            return int(str(v).strip())
        except Exception:
            return int(default)

    debug_nan = _env_flag("SLM_TRAIN_DEBUG_NAN", "1")
    skip_nonfinite_batch = _env_flag("SLM_TRAIN_SKIP_NONFINITE_BATCH", "1")
    clip_grad_norm = float(cfg.kd_params.get("clip_grad_norm", 1.0))
    try:
        clip_grad_norm = float(os.environ.get("SLM_TRAIN_CLIP_GRAD_NORM", clip_grad_norm))
    except Exception:
        pass
    log_every = max(1, _env_int("SLM_TRAIN_LOG_EVERY", 50))
    metrics: Dict[str, Any] = {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "learning_rate": lr,
        "loss_mean": None,
        "losses": [],
    }

    model.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"FT Teacher (LoRA) Epoch {epoch}")
        for batch in pbar:
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            with autocast_ctx(device):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                if loss is None:
                    # Defensive fallback: compute CE manually.
                    logits = out.logits
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                loss_to_backprop = loss / float(max(1, grad_accum_steps))

            # Early catch: if loss already NaN/Inf, don't backprop.
            if not torch.isfinite(loss.detach()).all():
                if debug_nan:
                    lr_now = None
                    try:
                        lr_now = float(lr_scheduler.get_last_lr()[0])
                    except Exception:
                        pass
                    print(f" (warn) SFT loss não-finita; epoch={epoch} step={step} lr={lr_now} loss={loss.detach().cpu()}")
                optimizer.zero_grad(set_to_none=True)
                if not skip_nonfinite_batch:
                    raise RuntimeError("SFT: loss não-finita detectada.")
                continue

            scaler.scale(loss_to_backprop).backward()

            if (step % grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm_val = float(grad_norm.detach().cpu().item())
                else:
                    grad_norm_val = float(grad_norm)
                if (not (grad_norm_val == grad_norm_val)) or (grad_norm_val == float("inf")):
                    if debug_nan:
                        print(f" (warn) SFT grad_norm não-finito; pulando step. epoch={epoch} step={step}")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            metrics["losses"].append(float(loss.detach().cpu().item()))
            if metrics["losses"]:
                recent = metrics["losses"][-min(50, len(metrics["losses"])) :]
                pbar.set_postfix({"loss": f"{sum(recent)/len(recent):.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

            if debug_nan and (step % log_every == 0):
                try:
                    print(f" (dbg) SFT epoch={epoch} step={step} loss={metrics['losses'][-1]:.6f}")
                except Exception:
                    pass

    if metrics["losses"]:
        metrics["loss_mean"] = float(sum(metrics["losses"]) / len(metrics["losses"]))

    return model, metrics


def _train_cascod_lora(
    cfg: EvidenceBasedConfig,
    model,
    tokenizer,
    ds_rationale,
    ds_answer,
    *,
    seed: int,
    alpha: float,
) -> tuple[Any, Dict[str, Any]]:
    """CasCoD training with teacher rationales.

    Implements:
      L = (1-α) L_rationale(q->r) + α L_answer(q,r->a)

    This is done via two forward passes per step (one for each objective).
    """

    import math

    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import get_scheduler

    set_seed(seed)
    device = cfg.device

    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("cascod_alpha must be in [0,1]")

    tok_r = preprocess_and_tokenize(ds_rationale, tokenizer, max_length=cfg.max_length)
    tok_a = preprocess_and_tokenize(ds_answer, tokenizer, max_length=cfg.max_length)

    # Stability + memory.
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Model should already have LoRA from stage1; if not, allow training anyway.
    model.train()
    model = safe_model_to(model, device)

    batch_size = int(cfg.kd_params.get("batch_size", 2))
    grad_accum_steps = int(cfg.kd_params.get("grad_accum_steps", 1))
    num_workers = int(cfg.kd_params.get("dataloader_num_workers", 0))

    # Critical scientific validity fix: rationale and answer batches must be aligned.
    # Two independently shuffled DataLoaders zipped together will mix examples.
    n_pairs = min(len(tok_r), len(tok_a))
    if len(tok_r) != len(tok_a):
        print(f"[WARN] CasCoD: tamanhos diferentes ds_rationale={len(tok_r)} ds_answer={len(tok_a)}; usando n_pairs={n_pairs}.")

    class _PairedCasCoDDataset(torch.utils.data.Dataset):
        def __init__(self, ds_r, ds_a, n: int):
            self.ds_r = ds_r
            self.ds_a = ds_a
            self.n = int(n)

        def __len__(self):
            return self.n

        def __getitem__(self, idx: int):
            r = self.ds_r[int(idx)]
            a = self.ds_a[int(idx)]
            return {
                "r_input_ids": r["input_ids"],
                "r_attention_mask": r.get("attention_mask"),
                "r_labels": r["labels"],
                "a_input_ids": a["input_ids"],
                "a_attention_mask": a.get("attention_mask"),
                "a_labels": a["labels"],
            }

    paired = _PairedCasCoDDataset(tok_r, tok_a, n_pairs)
    generator = torch.Generator()
    try:
        generator.manual_seed(int(seed))
    except Exception:
        pass
    dataloader = DataLoader(
        paired,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        generator=generator,
    )

    num_epochs = int(cfg.kd_params.get("epochs", 1))
    num_training_steps = max(1, num_epochs * len(dataloader))
    warmup_steps = max(0, int(0.1 * num_training_steps))

    lr = float(cfg.kd_params.get("learning_rates", {}).get("kd", 5e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = make_grad_scaler(device)

    def _env_flag(name: str, default: str = "0") -> bool:
        v = os.environ.get(name, default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        if v is None:
            return int(default)
        try:
            return int(str(v).strip())
        except Exception:
            return int(default)

    debug_nan = _env_flag("SLM_TRAIN_DEBUG_NAN", "1")
    skip_nonfinite_batch = _env_flag("SLM_TRAIN_SKIP_NONFINITE_BATCH", "1")
    clip_grad_norm = float(cfg.kd_params.get("clip_grad_norm", 1.0))
    try:
        clip_grad_norm = float(os.environ.get("SLM_TRAIN_CLIP_GRAD_NORM", clip_grad_norm))
    except Exception:
        pass
    log_every = max(1, _env_int("SLM_TRAIN_LOG_EVERY", 50))
    metrics: Dict[str, Any] = {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "learning_rate": lr,
        "alpha": alpha,
        "loss_mean": None,
        "losses": [],
        "losses_rationale": [],
        "losses_answer": [],
    }

    model.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"CasCoD Epoch {epoch}", total=len(dataloader))
        for batch in pbar:
            step += 1

            def _forward_loss(*, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], labels: torch.Tensor):
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                if loss is None:
                    logits = out.logits
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                return loss

            with autocast_ctx(device):
                loss_r = _forward_loss(
                    input_ids=batch["r_input_ids"],
                    attention_mask=batch.get("r_attention_mask"),
                    labels=batch["r_labels"],
                )
                loss_a = _forward_loss(
                    input_ids=batch["a_input_ids"],
                    attention_mask=batch.get("a_attention_mask"),
                    labels=batch["a_labels"],
                )
                loss = (1.0 - alpha) * loss_r + alpha * loss_a
                loss_to_backprop = loss / float(max(1, grad_accum_steps))

            if (not torch.isfinite(loss.detach()).all()) or (not torch.isfinite(loss_r.detach()).all()) or (not torch.isfinite(loss_a.detach()).all()):
                if debug_nan:
                    lr_now = None
                    try:
                        lr_now = float(lr_scheduler.get_last_lr()[0])
                    except Exception:
                        pass
                    print(
                        " (warn) CasCoD loss não-finita; "
                        f"epoch={epoch} step={step} lr={lr_now} "
                        f"loss={loss.detach().cpu()} loss_r={loss_r.detach().cpu()} loss_a={loss_a.detach().cpu()}"
                    )
                optimizer.zero_grad(set_to_none=True)
                if not skip_nonfinite_batch:
                    raise RuntimeError("CasCoD: loss não-finita detectada.")
                continue

            scaler.scale(loss_to_backprop).backward()

            if (step % grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm_val = float(grad_norm.detach().cpu().item())
                else:
                    grad_norm_val = float(grad_norm)
                if (not (grad_norm_val == grad_norm_val)) or (grad_norm_val == float("inf")):
                    if debug_nan:
                        print(f" (warn) CasCoD grad_norm não-finito; pulando step. epoch={epoch} step={step}")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            metrics["losses"].append(float(loss.detach().cpu().item()))
            metrics["losses_rationale"].append(float(loss_r.detach().cpu().item()))
            metrics["losses_answer"].append(float(loss_a.detach().cpu().item()))
            recent = metrics["losses"][-min(50, len(metrics["losses"])) :]
            pbar.set_postfix({"loss": f"{sum(recent)/len(recent):.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

            if debug_nan and (step % log_every == 0):
                try:
                    print(
                        f" (dbg) CasCoD epoch={epoch} step={step} loss={metrics['losses'][-1]:.6f} "
                        f"r={metrics['losses_rationale'][-1]:.6f} a={metrics['losses_answer'][-1]:.6f}"
                    )
                except Exception:
                    pass

    if metrics["losses"]:
        metrics["loss_mean"] = float(sum(metrics["losses"]) / len(metrics["losses"]))

    return model, metrics


def run_ft_teacher_baseline(
    cfg: EvidenceBasedConfig,
    *,
    enable_gsm8k_train: bool,
    enable_bbh_train: bool,
    eval_gsm8k: bool,
    eval_bbh: bool,
    eval_efficiency: bool,
    use_cot_prompt_eval: bool,
    eval_obqa: bool = False,
) -> Dict[str, Any]:
    """Baseline 0.1: fine-tune the teacher explicitly and evaluate it.

    Guarantees:
    - Training uses TRAIN split only (via load_training_dataset).
    - Saves a clear checkpoint (LoRA adapter + tokenizer).
    - Records the run in exp_id/state.json for traceability.
    """

    logger = ScientificLogger()
    evaluator = StandardizedEvaluator(cfg)

    flags = {
        "baseline": "ft_teacher",
        "enable_gsm8k_train": enable_gsm8k_train,
        "enable_bbh_train": enable_bbh_train,
        "eval_gsm8k": eval_gsm8k,
        "eval_bbh": eval_bbh,
        "eval_efficiency": eval_efficiency,
        "use_cot_prompt_eval": use_cot_prompt_eval,
        "eval_obqa": bool(eval_obqa),
    }

    exp_id = _experiment_id(cfg, kd_modes=["ft_teacher"], flags=flags)
    exp_dir = cfg.experiments_dir / exp_id
    state_path = exp_dir / "state.json"
    state = _load_state(state_path)

    logger.log_phase("EXPERIMENT", {"id": exp_id, "dir": str(exp_dir), "flags": flags})
    logger.log_hyperparameters(cfg.to_metadata())

    results: Dict[str, Any] = {"metadata": cfg.to_metadata(), "conditions": {}}
    cond_name = "ft_teacher"
    cond_runs: List[Dict[str, Any]] = []

    base_teacher_name = str(cfg.model_hierarchy.get("teacher_medium"))

    for seed in cfg.seeds:
        run_key = f"{cond_name}_seed{seed}"
        if state.get("completed", {}).get(run_key):
            print(f" Pulando run já completado: {run_key}")
            cond_runs.append(state["completed"][run_key])
            continue

        print(f"\n Baseline FT Teacher | seed={seed}")
        set_seed(seed)

        # TRAIN split only.
        train_ds = load_training_dataset(enable_gsm8k_train, enable_bbh_train, cfg.train_limit, seed=seed)

        teacher_model, teacher_tok = setup_model_and_tokenizer(base_teacher_name, cfg.device, cfg.quantization)
        teacher_model, train_metrics = _train_sft_lora(cfg, teacher_model, teacher_tok, train_ds, seed=seed)

        # Save immediately.
        save_dir = cfg.models_dir / exp_id / f"ft_teacher_seed{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        teacher_model.save_pretrained(save_dir)
        teacher_tok.save_pretrained(save_dir)

        state.setdefault("artifacts", {})[f"ft_teacher_seed{seed}_dir"] = str(save_dir)
        state.setdefault("artifacts", {})["ft_teacher_base_model"] = base_teacher_name
        _save_state(state_path, state)
        print(f" Teacher (adapter) salvo em: {save_dir}")

        # Eval (kept consistent with the repo evaluator).
        eval_results = evaluator.evaluate(
            teacher_model,
            teacher_tok,
            seed=seed,
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_obqa=bool(eval_obqa),
            eval_efficiency=eval_efficiency,
            use_cot_prompt=use_cot_prompt_eval,
            generation_cfg=cfg.eval_generation,
        )

        run_payload = {
            "seed": seed,
            "condition": cond_name,
            "description": "Fine-tuning do teacher (LoRA/Colab-friendly)",
            "training": train_metrics,
            "evaluation": eval_results,
            "artifacts": {"teacher_dir": str(save_dir), "teacher_base": base_teacher_name},
        }

        state.setdefault("completed", {})[run_key] = run_payload
        _save_state(state_path, state)
        cond_runs.append(run_payload)

        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["conditions"][cond_name] = {"description": "FT Teacher", "runs": cond_runs}

    # Reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_json = cfg.reports_dir / f"comprehensive_report_{exp_id}_{timestamp}.json"
    summary_txt = cfg.reports_dir / f"results_summary_{exp_id}_{timestamp}.txt"

    results.setdefault("artifacts", {})
    results["artifacts"].update(
        {
            "experiment_id": exp_id,
            "experiment_dir": str(exp_dir),
            "report_json": str(report_json),
            "summary_txt": str(summary_txt),
        }
    )

    write_report_json(report_json, results)
    write_summary_txt(summary_txt, results)

    plot_paths = write_plots(cfg.reports_dir, results, prefix=f"plots_{exp_id}_{timestamp}")
    results["artifacts"]["plots"] = [str(p) for p in plot_paths]
    write_report_json(report_json, results)

    print(" Relatórios salvos:")
    print(f"   - JSON: {report_json}")
    print(f"   - TXT:  {summary_txt}")
    if plot_paths:
        print("   - PLOTS:")
        for p in plot_paths:
            print(f"     * {p}")

    return results


def run_ft_student_baseline(
    cfg: EvidenceBasedConfig,
    *,
    student_key: str,
    enable_gsm8k_train: bool,
    enable_bbh_train: bool,
    eval_gsm8k: bool,
    eval_bbh: bool,
    eval_efficiency: bool,
    use_cot_prompt_eval: bool,
    eval_obqa: bool = False,
) -> Dict[str, Any]:
    """Baseline 0.2: fine-tune the student explicitly (no KD) and evaluate.

    Guarantees:
    - Training uses TRAIN split only (via load_training_dataset).
    - Does NOT reuse teacher CoT/logits caches (no KD components invoked).
    - Saves a clear checkpoint (LoRA adapter + tokenizer).
    - Records the run in exp_id/state.json for traceability.
    """

    logger = ScientificLogger()
    evaluator = StandardizedEvaluator(cfg)

    flags = {
        "baseline": "ft_student",
        "student_key": str(student_key),
        "enable_gsm8k_train": enable_gsm8k_train,
        "enable_bbh_train": enable_bbh_train,
        "eval_gsm8k": eval_gsm8k,
        "eval_bbh": eval_bbh,
        "eval_efficiency": eval_efficiency,
        "use_cot_prompt_eval": use_cot_prompt_eval,
        "eval_obqa": bool(eval_obqa),
    }

    exp_id = _experiment_id(cfg, kd_modes=["ft_student"], flags=flags)
    exp_dir = cfg.experiments_dir / exp_id
    state_path = exp_dir / "state.json"
    state = _load_state(state_path)

    logger.log_phase("EXPERIMENT", {"id": exp_id, "dir": str(exp_dir), "flags": flags})
    logger.log_hyperparameters(cfg.to_metadata())

    results: Dict[str, Any] = {"metadata": cfg.to_metadata(), "conditions": {}}
    cond_name = "ft_student"
    cond_runs: List[Dict[str, Any]] = []

    if student_key not in cfg.model_hierarchy:
        raise ValueError(f"student_key inválido: {student_key}. Opções: {sorted(cfg.model_hierarchy.keys())}")
    base_student_name = str(cfg.model_hierarchy.get(student_key))

    for seed in cfg.seeds:
        run_key = f"{cond_name}_seed{seed}"
        if state.get("completed", {}).get(run_key):
            print(f" Pulando run já completado: {run_key}")
            cond_runs.append(state["completed"][run_key])
            continue

        print(f"\n Baseline FT Student | student={student_key} | seed={seed}")
        set_seed(seed)

        # TRAIN split only.
        train_ds = load_training_dataset(enable_gsm8k_train, enable_bbh_train, cfg.train_limit, seed=seed)

        student_model, student_tok = setup_model_and_tokenizer(base_student_name, cfg.device, cfg.quantization)
        student_model, train_metrics = _train_sft_lora(cfg, student_model, student_tok, train_ds, seed=seed)

        # Save immediately.
        save_dir = cfg.models_dir / exp_id / f"ft_student_seed{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        student_model.save_pretrained(save_dir)
        student_tok.save_pretrained(save_dir)

        state.setdefault("artifacts", {})[f"ft_student_seed{seed}_dir"] = str(save_dir)
        state.setdefault("artifacts", {})["ft_student_base_model"] = base_student_name
        state.setdefault("artifacts", {})["ft_student_key"] = str(student_key)
        _save_state(state_path, state)
        print(f" Student (adapter) salvo em: {save_dir}")

        eval_results = evaluator.evaluate(
            student_model,
            student_tok,
            seed=seed,
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_obqa=bool(eval_obqa),
            eval_efficiency=eval_efficiency,
            use_cot_prompt=use_cot_prompt_eval,
            generation_cfg=cfg.eval_generation,
        )

        run_payload = {
            "seed": seed,
            "condition": cond_name,
            "description": f"Fine-tuning do student ({student_key}) sem KD (LoRA/Colab-friendly)",
            "training": train_metrics,
            "evaluation": eval_results,
            "artifacts": {"student_dir": str(save_dir), "student_base": base_student_name, "student_key": str(student_key)},
        }

        state.setdefault("completed", {})[run_key] = run_payload
        _save_state(state_path, state)
        cond_runs.append(run_payload)

        del student_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["conditions"][cond_name] = {"description": "FT Student", "runs": cond_runs}

    # Reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_json = cfg.reports_dir / f"comprehensive_report_{exp_id}_{timestamp}.json"
    summary_txt = cfg.reports_dir / f"results_summary_{exp_id}_{timestamp}.txt"

    results.setdefault("artifacts", {})
    results["artifacts"].update(
        {
            "experiment_id": exp_id,
            "experiment_dir": str(exp_dir),
            "report_json": str(report_json),
            "summary_txt": str(summary_txt),
        }
    )

    write_report_json(report_json, results)
    write_summary_txt(summary_txt, results)

    plot_paths = write_plots(cfg.reports_dir, results, prefix=f"plots_{exp_id}_{timestamp}")
    results["artifacts"]["plots"] = [str(p) for p in plot_paths]
    write_report_json(report_json, results)

    print(" Relatórios salvos:")
    print(f"   - JSON: {report_json}")
    print(f"   - TXT:  {summary_txt}")
    if plot_paths:
        print("   - PLOTS:")
        for p in plot_paths:
            print(f"     * {p}")

    return results


def _load_model_and_tokenizer_from_dir(model_dir: Path, device: torch.device, quant_cfg: Dict[str, Any]):
    """Load either a full HF model dir (config.json) or a PEFT adapter dir (adapter_config.json).

    This mirrors evaluate_saved_models.py but keeps run_experiment self-contained.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.model_max_length = get_safe_tokenizer_length(tokenizer, fallback=2048, upper_bound=4096)
    tokenizer_len = len(tokenizer)

    model_kwargs: Dict[str, Any] = {}
    if bool(quant_cfg.get("load_in_4bit")):
        compute_dtype = quant_cfg.get("bnb_4bit_compute_dtype", torch.bfloat16)
        if isinstance(compute_dtype, str):
            compute_dtype = getattr(torch, compute_dtype, torch.bfloat16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = quant_cfg.get("device_map", "auto")

    if (model_dir / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        if not bool(quant_cfg.get("load_in_4bit")):
            model = safe_model_to(model, device)
        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()
        # Unique identity for caches.
        try:
            setattr(model, "name_or_path", str(model_dir))
        except Exception:
            pass
        return model, tokenizer

    if (model_dir / "adapter_config.json").exists():
        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
        if not base_name:
            raise ValueError(
                f"adapter_config.json em {model_dir} não contém 'base_model_name_or_path'. "
                "Não é possível carregar o modelo base."
            )

        base_model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)

        # Ensure embedding size matches tokenizer before adapter load.
        try:
            emb = base_model.get_input_embeddings()
            if emb is not None and tokenizer_len > int(emb.weight.shape[0]):
                base_model.resize_token_embeddings(tokenizer_len)
        except Exception:
            pass

        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, model_dir)
        except Exception as exc:
            raise RuntimeError(
                "Falha ao carregar adapter PEFT (instale 'peft' e verifique compatibilidade). "
                f"Erro: {exc}"
            )

        if not bool(quant_cfg.get("load_in_4bit")):
            model = safe_model_to(model, device)
        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()

        # Make teacher identity unique so caches don't collide with base teacher.
        unique_id = f"{base_name}::adapter::{str(model_dir)}"
        try:
            setattr(model, "name_or_path", unique_id)
        except Exception:
            pass

        return model, tokenizer

    raise ValueError(f"Diretório não parece ser um modelo HF nem um adapter PEFT: {model_dir}")


def run_kd_logits_baseline(
    cfg: EvidenceBasedConfig,
    *,
    teacher_ckpt_dir: str,
    student_key: str,
    enable_gsm8k_train: bool,
    enable_bbh_train: bool,
    eval_gsm8k: bool,
    eval_bbh: bool,
    eval_efficiency: bool,
    use_cot_prompt_eval: bool,
    eval_obqa: bool = False,
) -> Dict[str, Any]:
    """Baseline 0.3: logits-KD using an ADJUSTED teacher -> student.

    Requirements:
    - Enforce teacher/student tokenizer compatibility.
    - Cache logits separated by teacher identity + tokenizer hash + split + params.
    - Logits must be computed on TRAIN split only (no eval leakage).
    """

    logger = ScientificLogger()
    evaluator = StandardizedEvaluator(cfg)

    teacher_dir = Path(teacher_ckpt_dir)
    if not teacher_dir.exists():
        raise ValueError(f"teacher_ckpt_dir não existe: {teacher_dir}")

    flags = {
        "baseline": "kd_logits",
        "teacher_ckpt_dir": str(teacher_dir),
        "student_key": str(student_key),
        "enable_gsm8k_train": enable_gsm8k_train,
        "enable_bbh_train": enable_bbh_train,
        "eval_gsm8k": eval_gsm8k,
        "eval_bbh": eval_bbh,
        "eval_efficiency": eval_efficiency,
        "use_cot_prompt_eval": use_cot_prompt_eval,
        "eval_obqa": bool(eval_obqa),
    }

    exp_id = _experiment_id(cfg, kd_modes=["kd_logits"], flags=flags)
    exp_dir = cfg.experiments_dir / exp_id
    state_path = exp_dir / "state.json"
    state = _load_state(state_path)

    logger.log_phase("EXPERIMENT", {"id": exp_id, "dir": str(exp_dir), "flags": flags})
    logger.log_hyperparameters(cfg.to_metadata())

    results: Dict[str, Any] = {"metadata": cfg.to_metadata(), "conditions": {}}
    cond_name = "kd_logits"
    cond_runs: List[Dict[str, Any]] = []

    if student_key not in cfg.model_hierarchy:
        raise ValueError(f"student_key inválido: {student_key}. Opções: {sorted(cfg.model_hierarchy.keys())}")
    student_name = str(cfg.model_hierarchy[student_key])

    for seed in cfg.seeds:
        run_key = f"{cond_name}_seed{seed}"
        if state.get("completed", {}).get(run_key):
            print(f" Pulando run já completado: {run_key}")
            cond_runs.append(state["completed"][run_key])
            continue

        print(f"\n Baseline KD Logits | teacher_ckpt={teacher_dir} | student={student_key} | seed={seed}")
        set_seed(seed)

        # TRAIN split only.
        train_ds = load_training_dataset(enable_gsm8k_train, enable_bbh_train, cfg.train_limit, seed=seed)

        # Load adjusted teacher from checkpoint directory (adapter/full model).
        teacher_model, teacher_tok = _load_model_and_tokenizer_from_dir(teacher_dir, cfg.device, cfg.quantization)

        # Load student (fresh).
        student_model, student_tok = setup_model_and_tokenizer(student_name, cfg.device, cfg.quantization)

        # Block if tokenizers incompatible (scientific validity requirement).
        assert_tokenizer_compatible_for_logits_kd(
            teacher_tok,
            student_tok,
            context=f"baseline=kd_logits, teacher_ckpt={teacher_dir}, student={student_key}",
        )

        # Cache teacher logits (versioned by teacher identity + tokenizer hash + dataset fp + split + params).
        logits_dir = cache_teacher_logits(
            teacher_model,
            teacher_tok,
            train_ds,
            cache_root=cfg.cache_dir,
            batch_size=cfg.kd_params["batch_size"],
            device=str(cfg.device),
            generation_cfg=GenerationConfig(max_new_tokens=0, temperature=0.0, do_sample=False),
            split="train",
            seed=seed,
            kd_mode="traditional",
            input_kind="prompt_completion",
            train_limit=cfg.train_limit,
            max_length=cfg.max_length,
        )

        # Free teacher memory before training.
        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Distill student using cache only.
        # Optional: save intermediate checkpoints per epoch for postmortem debugging.
        if str(os.environ.get("SLM_ENABLE_KD_EPOCH_CKPTS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}:
            ckpt_dir = cfg.models_dir / exp_id / f"kd_logits_seed{seed}_ckpts"
            os.environ["SLM_KD_CKPT_DIR"] = str(ckpt_dir)
            os.environ["SLM_KD_SAVE_EVERY_EPOCH"] = "1"

        distiller = TraditionalKDDistiller(cfg, cache_dir=str(logits_dir))
        trained_model, train_metrics = distiller.distill(
            student_model,
            teacher_model=None,
            student_tokenizer=student_tok,
            raw_dataset=train_ds,
            seed=seed,
            use_cache=True,
        )

        # Save.
        save_dir = cfg.models_dir / exp_id / f"kd_logits_seed{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        trained_model.save_pretrained(save_dir)
        student_tok.save_pretrained(save_dir)
        print(f" Student KD-logits salvo em: {save_dir}")

        # Eval.
        eval_results = evaluator.evaluate(
            trained_model,
            student_tok,
            seed=seed,
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_obqa=bool(eval_obqa),
            eval_efficiency=eval_efficiency,
            use_cot_prompt=use_cot_prompt_eval,
            generation_cfg=cfg.eval_generation,
        )

        run_payload = {
            "seed": seed,
            "condition": cond_name,
            "description": "KD por logits (teacher ajustado → student)",
            "training": train_metrics,
            "evaluation": eval_results,
            "artifacts": {
                "teacher_ckpt_dir": str(teacher_dir),
                "logits_cache_dir": str(logits_dir),
                "student_dir": str(save_dir),
                "student_key": str(student_key),
            },
        }

        state.setdefault("completed", {})[run_key] = run_payload
        state.setdefault("artifacts", {})[f"kd_logits_seed{seed}_dir"] = str(save_dir)
        state.setdefault("artifacts", {})[f"kd_logits_seed{seed}_logits_cache"] = str(logits_dir)
        state.setdefault("artifacts", {})["kd_logits_teacher_ckpt_dir"] = str(teacher_dir)
        _save_state(state_path, state)
        cond_runs.append(run_payload)

        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["conditions"][cond_name] = {"description": "KD logits baseline", "runs": cond_runs}

    # Reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_json = cfg.reports_dir / f"comprehensive_report_{exp_id}_{timestamp}.json"
    summary_txt = cfg.reports_dir / f"results_summary_{exp_id}_{timestamp}.txt"

    results.setdefault("artifacts", {})
    results["artifacts"].update(
        {
            "experiment_id": exp_id,
            "experiment_dir": str(exp_dir),
            "report_json": str(report_json),
            "summary_txt": str(summary_txt),
        }
    )

    write_report_json(report_json, results)
    write_summary_txt(summary_txt, results)

    plot_paths = write_plots(cfg.reports_dir, results, prefix=f"plots_{exp_id}_{timestamp}")
    results["artifacts"]["plots"] = [str(p) for p in plot_paths]
    write_report_json(report_json, results)

    print(" Relatórios salvos:")
    print(f"   - JSON: {report_json}")
    print(f"   - TXT:  {summary_txt}")
    if plot_paths:
        print("   - PLOTS:")
        for p in plot_paths:
            print(f"     * {p}")

    return results


def run_kd_cot_standard_baseline(
    cfg: EvidenceBasedConfig,
    *,
    teacher_ckpt_dir: Optional[str],
    student_key: str,
    enable_gsm8k_train: bool,
    enable_bbh_train: bool,
    eval_gsm8k: bool,
    eval_bbh: bool,
    eval_efficiency: bool,
    use_cot_prompt_eval: bool,
    eval_obqa: bool = False,
    granularity_level: int = 0,
) -> Dict[str, Any]:
    """Baseline 0.4: KD CoT padrão (text-only), sem otimizações.

    Definition enforced:
    - Teacher generates CoT BEFORE the answer.
    - Student trains with CE on teacher-generated sequences (no logits-KD).
    - Prompt tokens (question/instruction) are masked out; loss applies only to CoT+answer.
    - CoT is generated from TRAIN split only (no eval leakage).
    """

    logger = ScientificLogger()
    evaluator = StandardizedEvaluator(cfg)

    if student_key not in cfg.model_hierarchy:
        raise ValueError(f"student_key inválido: {student_key}. Opções: {sorted(cfg.model_hierarchy.keys())}")
    student_name = str(cfg.model_hierarchy[student_key])

    teacher_dir = Path(teacher_ckpt_dir) if teacher_ckpt_dir else None
    if teacher_dir is not None and not teacher_dir.exists():
        raise ValueError(f"teacher_ckpt_dir não existe: {teacher_dir}")

    flags = {
        "baseline": "kd_cot_standard",
        "teacher_ckpt_dir": str(teacher_dir) if teacher_dir else None,
        "student_key": str(student_key),
        "enable_gsm8k_train": enable_gsm8k_train,
        "enable_bbh_train": enable_bbh_train,
        "eval_gsm8k": eval_gsm8k,
        "eval_bbh": eval_bbh,
        "eval_efficiency": eval_efficiency,
        "use_cot_prompt_eval": use_cot_prompt_eval,
        "eval_obqa": bool(eval_obqa),
        "granularity_level": int(granularity_level or 0),
        "teacher_cot_generation": cfg.teacher_cot_generation.to_jsonable(),
    }

    exp_id = _experiment_id(cfg, kd_modes=["kd_cot_standard"], flags=flags)
    exp_dir = cfg.experiments_dir / exp_id
    state_path = exp_dir / "state.json"
    state = _load_state(state_path)

    logger.log_phase("EXPERIMENT", {"id": exp_id, "dir": str(exp_dir), "flags": flags})
    logger.log_hyperparameters(cfg.to_metadata())

    results: Dict[str, Any] = {"metadata": cfg.to_metadata(), "conditions": {}}
    cond_name = "kd_cot_standard"
    cond_runs: List[Dict[str, Any]] = []

    for seed in cfg.seeds:
        run_key = f"{cond_name}_seed{seed}"
        if state.get("completed", {}).get(run_key):
            print(f" Pulando run já completado: {run_key}")
            cond_runs.append(state["completed"][run_key])
            continue

        print(f"\n Baseline KD CoT padrão | student={student_key} | seed={seed}")
        set_seed(seed)

        # TRAIN split only.
        train_ds = load_training_dataset(enable_gsm8k_train, enable_bbh_train, cfg.train_limit, seed=seed)

        # Load teacher for CoT generation (adjusted checkpoint if provided; else base teacher).
        if teacher_dir is not None:
            teacher_model, teacher_tok = _load_model_and_tokenizer_from_dir(teacher_dir, cfg.device, cfg.quantization)
            teacher_identity = str(teacher_dir)
        else:
            teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)
            teacher_identity = str(cfg.model_hierarchy["teacher_medium"])

        # Generate (or reuse) CoT cache from TRAIN only.
        cot_file = cache_teacher_cot(
            teacher_model,
            teacher_tok,
            train_ds,
            cache_root=cfg.cache_dir,
            batch_size=max(1, cfg.kd_params["batch_size"] // 2),
            device=str(cfg.device),
            generation_cfg=cfg.teacher_cot_generation,
            split="train",
            seed=seed,
            prompt_max_length=cfg.max_length,
            train_limit=cfg.train_limit,
            granularity_level=int(granularity_level or 0),
        )

        # Free teacher memory; baseline is text-only distillation.
        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        student_model, student_tok = setup_model_and_tokenizer(student_name, cfg.device, cfg.quantization)

        # Distill (CE-only): teacher logits are NOT used.
        distiller = ReasoningAwareDistiller(cfg, cot_cache_path=str(cot_file), logits_cache_dir=None)
        trained_model, train_metrics = distiller.distill_with_reasoning(
            student_model,
            teacher_model=None,
            student_tokenizer=student_tok,
            raw_dataset=train_ds,
            seed=seed,
            use_cot_cache=True,
            use_logits_cache=False,
            use_teacher_logits=False,
            granularity_level=int(granularity_level or 0),
        )

        save_dir = cfg.models_dir / exp_id / f"kd_cot_standard_seed{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        trained_model.save_pretrained(save_dir)
        student_tok.save_pretrained(save_dir)
        print(f" Student KD-CoT (padrão) salvo em: {save_dir}")

        eval_results = evaluator.evaluate(
            trained_model,
            student_tok,
            seed=seed,
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_obqa=bool(eval_obqa),
            eval_efficiency=eval_efficiency,
            use_cot_prompt=use_cot_prompt_eval,
            generation_cfg=cfg.eval_generation,
        )

        run_payload = {
            "seed": seed,
            "condition": cond_name,
            "description": "KD CoT padrão (texto; CoT→resposta; CE-only)",
            "training": train_metrics,
            "evaluation": eval_results,
            "artifacts": {
                "teacher": teacher_identity,
                "teacher_ckpt_dir": str(teacher_dir) if teacher_dir else None,
                "cot_cache_file": str(cot_file),
                "student_dir": str(save_dir),
                "student_key": str(student_key),
            },
        }

        state.setdefault("completed", {})[run_key] = run_payload
        state.setdefault("artifacts", {})[f"kd_cot_standard_seed{seed}_dir"] = str(save_dir)
        state.setdefault("artifacts", {})[f"kd_cot_standard_seed{seed}_cot_cache"] = str(cot_file)
        _save_state(state_path, state)
        cond_runs.append(run_payload)

        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["conditions"][cond_name] = {"description": "KD CoT padrão", "runs": cond_runs}

    # Reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_json = cfg.reports_dir / f"comprehensive_report_{exp_id}_{timestamp}.json"
    summary_txt = cfg.reports_dir / f"results_summary_{exp_id}_{timestamp}.txt"

    results.setdefault("artifacts", {})
    results["artifacts"].update(
        {
            "experiment_id": exp_id,
            "experiment_dir": str(exp_dir),
            "report_json": str(report_json),
            "summary_txt": str(summary_txt),
        }
    )

    write_report_json(report_json, results)
    write_summary_txt(summary_txt, results)

    plot_paths = write_plots(cfg.reports_dir, results, prefix=f"plots_{exp_id}_{timestamp}")
    results["artifacts"]["plots"] = [str(p) for p in plot_paths]
    write_report_json(report_json, results)

    print(" Relatórios salvos:")
    print(f"   - JSON: {report_json}")
    print(f"   - TXT:  {summary_txt}")
    if plot_paths:
        print("   - PLOTS:")
        for p in plot_paths:
            print(f"     * {p}")

    return results


def _tokenizer_compat_key(tokenizer) -> str:
    # Single source of truth (shared with cache versioning).
    return tokenizer_fingerprint(tokenizer)


def assert_tokenizer_compatible_for_logits_kd(teacher_tokenizer, student_tokenizer, *, context: str) -> None:
    t = _tokenizer_compat_key(teacher_tokenizer)
    s = _tokenizer_compat_key(student_tokenizer)
    if not t or not s or t != s:
        raise ValueError(
            "Configuração cientificamente inválida para logits-KD: "
            f"tokenizers incompatíveis ({context}). "
            f"teacher_tokenizer_hash='{t}', student_tokenizer_hash='{s}'. "
            f"teacher_tokenizer_name='{getattr(teacher_tokenizer, 'name_or_path', None)}', "
            f"student_tokenizer_name='{getattr(student_tokenizer, 'name_or_path', None)}'. "
            "Use student compatível com o teacher, ou desative logits-KD (--no_logits_kd)."
        )


def setup_model_and_tokenizer(model_name: str, device: torch.device, quant_cfg: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = get_safe_tokenizer_length(tokenizer, fallback=2048, upper_bound=4096)

    model_kwargs: Dict[str, Any] = {}
    if bool(quant_cfg.get("load_in_4bit")):
        compute_dtype = quant_cfg.get("bnb_4bit_compute_dtype", torch.bfloat16)
        if isinstance(compute_dtype, str):
            compute_dtype = getattr(torch, compute_dtype, torch.bfloat16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = quant_cfg.get("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not bool(quant_cfg.get("load_in_4bit")):
        model = safe_model_to(model, device)
    ensure_tokenizer_has_pad(tokenizer, model)
    return model, tokenizer


def _experiment_id(cfg: EvidenceBasedConfig, kd_modes: Sequence[str], flags: Dict[str, Any]) -> str:
    # Scientific traceability fix: include all variables that change the actual
    # experimental outcomes to avoid mixing runs in the same state.json.
    key = {
        "models": cfg.model_hierarchy,
        "max_length": cfg.max_length,
        "train_limit": cfg.train_limit,
        "eval_limits": {
            "gsm8k": cfg.eval_limit_gsm8k,
            "bbh": cfg.eval_limit_bbh,
        },
        "seeds": list(cfg.seeds),
        "kd_params": cfg.kd_params,
        "quantization": cfg.quantization,
        "kd_modes": list(kd_modes),
        "flags": flags,
        "teacher_cot_generation": cfg.teacher_cot_generation.to_jsonable(),
        "eval_generation": cfg.eval_generation.to_jsonable(),
    }
    import hashlib

    blob = json.dumps(key, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"completed": {}}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": {}}


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run_experiment(
    cfg: EvidenceBasedConfig,
    kd_modes: Sequence[str],
    enable_gsm8k_train: bool,
    enable_bbh_train: bool,
    eval_gsm8k: bool,
    eval_bbh: bool,
    eval_obqa: bool,
    eval_efficiency: bool,
    use_cot_prompt_eval: bool,
    prepare_caches: bool,
    cache_logits: bool,
    cache_cot: bool,
    use_logits_kd: bool,
    allow_insufficient_runs: bool,
    hypothesis_metric: str,
    student_key: str,
    granularity_level: int = 0,
    granularity_one_shot: bool = False,
    granularity_multi_level: bool = False,
    cascod_stage1_epochs: int = 1,
    cascod_stage2_epochs: int = -1,
    cascod_alpha: float = 0.3,
    post_cot: bool = False,
    post_cot_gold_rationale: bool = False,
    post_cot_use_ig: bool = False,
    post_cot_ig_steps: int = 8,
    post_cot_ig_top_frac: float = 0.3,
    cascod_filter_by_gold: bool = True,
) -> Dict[str, Any]:
    logger = ScientificLogger()
    analyst = StatisticalAnalyst(alpha=cfg.alpha_level)
    evaluator = StandardizedEvaluator(cfg)

    # Reproducibility controls (optional).
    deterministic = _env_flag("SLM_DETERMINISTIC", "0")
    if bool(deterministic):
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    combo_d_enabled = "combo_d" in set(kd_modes or [])

    def _cot_variant_key(
        *,
        post: bool,
        gran: int,
        one_shot: bool,
        multi_level: bool,
        gold_rationale: bool,
        ig: bool,
        filter_gold: bool,
    ) -> str:
        return (
            f"post_cot={int(bool(post))}|"
            f"granularity={int(gran or 0)}|"
            f"one_shot={int(bool(one_shot))}|"
            f"multi_level={int(bool(multi_level))}|"
            f"gold_rationale={int(bool(gold_rationale))}|"
            f"ig={int(bool(ig))}|"
            f"filter_gold={int(bool(filter_gold))}"
        )

    flags = {
        "enable_gsm8k_train": enable_gsm8k_train,
        "enable_bbh_train": enable_bbh_train,
        "eval_gsm8k": eval_gsm8k,
        "eval_bbh": eval_bbh,
        "eval_obqa": bool(eval_obqa),
        "eval_efficiency": eval_efficiency,
        "use_cot_prompt_eval": use_cot_prompt_eval,
        "prepare_caches": prepare_caches,
        "cache_logits": cache_logits,
        "cache_cot": cache_cot,
        "use_logits_kd": use_logits_kd,
        "allow_insufficient_runs": allow_insufficient_runs,
        "hypothesis_metric": hypothesis_metric,
        "student_key": student_key,
        "granularity_level": int(granularity_level or 0),
        "granularity_one_shot": bool(granularity_one_shot),
        "granularity_multi_level": bool(granularity_multi_level),
        "cascod_stage1_epochs": int(cascod_stage1_epochs or 0),
        "cascod_stage2_epochs": int(cascod_stage2_epochs or -1),
        "cascod_alpha": float(cascod_alpha),
        "post_cot": bool(post_cot),
        "post_cot_gold_rationale": bool(post_cot_gold_rationale),
        "post_cot_use_ig": bool(post_cot_use_ig),
        "post_cot_ig_steps": int(post_cot_ig_steps or 0),
        "post_cot_ig_top_frac": float(post_cot_ig_top_frac or 0.0),
        "cascod_filter_by_gold": bool(cascod_filter_by_gold),
        "combo_d_enabled": bool(combo_d_enabled),
        "deterministic": bool(deterministic),
    }

    exp_id = _experiment_id(cfg, kd_modes, flags)
    exp_dir = cfg.experiments_dir / exp_id
    state_path = exp_dir / "state.json"
    state = _load_state(state_path)

    logger.log_phase("EXPERIMENT", {"id": exp_id, "dir": str(exp_dir), "flags": flags})
    logger.log_hyperparameters(cfg.to_metadata())
    logger.log_phase("ENVIRONMENT", _collect_environment_metadata())

    conditions: Dict[str, Dict[str, Any]] = {}
    if "traditional" in kd_modes:
        conditions["kd_traditional"] = {"mode": "traditional", "description": "KD tradicional (answer-only)"}
    if "reasoning" in kd_modes:
        conditions["kd_with_reasoning"] = {"mode": "reasoning", "description": "KD com raciocínio (CoT-aware)"}
    if "cascod" in kd_modes:
        conditions["kd_cascod"] = {
            "mode": "cascod",
            "description": "CasCoD (2B): q→r e q,r→a com L=(1-α)Lr+αLa (teacher rationales)",
        }
    if "combo_d" in kd_modes:
        conditions["kd_combo_d"] = {
            "mode": "cascod",
            "description": "Condição D (A+B+C): granularity + CasCoD + Post-CoT",
            "force_post_cot": True,
        }
    if not conditions:
        raise ValueError("Nenhum modo de KD selecionado.")

    # Prepare datasets once per seed (training)
    results: Dict[str, Any] = {"metadata": cfg.to_metadata(), "conditions": {}, "eval_protocols": {}}
    eval_protocol_strict = str(os.environ.get("SLM_EVAL_PROTOCOL_STRICT", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
    results["metadata"].setdefault("environment", _collect_environment_metadata())
    results["metadata"].setdefault("reproducibility", {})
    results["metadata"]["reproducibility"].update(
        {
            "deterministic": bool(deterministic),
        }
    )
    results["metadata"].setdefault("controls", {})
    results["metadata"]["controls"].update(
        {
            "eval_protocol_strict": bool(eval_protocol_strict),
            "logits_cache_load_mode": str(os.environ.get("SLM_LOGITS_CACHE_LOAD_MODE", "auto")),
            "logits_cache_eager_max_mb": float(os.environ.get("SLM_LOGITS_CACHE_EAGER_MAX_MB", "2048")),
            "reasoning_mask_strict": _env_flag("SLM_REASONING_MASK_STRICT", "0"),
            "reasoning_mask_fallback_to_completion": _env_flag("SLM_REASONING_MASK_FALLBACK_TO_COMPLETION", "1"),
            "train_sanitize_logits": _env_flag("SLM_TRAIN_SANITIZE_LOGITS", "1"),
            "train_max_logit_abs": float(os.environ.get("SLM_TRAIN_MAX_LOGIT_ABS", "100.0")),
            "cache_sanitize_logits": str(os.environ.get("SLM_CACHE_SANITIZE_LOGITS", "1")),
            "cache_clamp_fp16": str(os.environ.get("SLM_CACHE_CLAMP_FP16", "1")),
            "cache_fp16_safe_abs": float(os.environ.get("SLM_CACHE_FP16_SAFE_ABS", "60000.0")),
        }
    )

    # Optional cache preparation
    if prepare_caches:
        # Prepare caches against the *training* split only (leakage control is
        # handled in data/eval modules via explicit splitting).
        train_ds_for_cache = load_training_dataset(enable_gsm8k_train, enable_bbh_train, cfg.train_limit, seed=cfg.seeds[0])

        teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)

        if cache_cot:
            use_one_shot = bool(granularity_one_shot) and int(granularity_level or 0) > 0

            # Prepare the CoT variants actually needed by selected modes.
            variants = []
            if "reasoning" in kd_modes:
                variants.append({"post": bool(post_cot), "filter_gold": False})
            if "cascod" in kd_modes:
                variants.append({"post": bool(post_cot), "filter_gold": bool(cascod_filter_by_gold)})
            if combo_d_enabled:
                variants.append({"post": True, "filter_gold": bool(cascod_filter_by_gold)})
            if not variants:
                variants.append({"post": bool(post_cot), "filter_gold": False})

            tcf = state.get("teacher_cot_files")
            if not isinstance(tcf, dict):
                tcf = {}

            cot_file_for_legacy = None
            for v in variants:
                v_post = bool(v["post"])
                v_filter = bool(v["filter_gold"])
                v_gold_rationale = bool(post_cot_gold_rationale) and v_post
                cot_file = cache_teacher_cot(
                    teacher_model,
                    teacher_tok,
                    train_ds_for_cache,
                    cache_root=cfg.cache_dir,
                    batch_size=max(1, cfg.kd_params["batch_size"] // 2),
                    device=str(cfg.device),
                    generation_cfg=cfg.teacher_cot_generation,
                    split="train",
                    seed=cfg.seeds[0],
                    prompt_max_length=cfg.max_length,
                    train_limit=cfg.train_limit,
                    granularity_level=int(granularity_level or 0),
                    granularity_multi_level=bool(granularity_multi_level),
                    post_cot=bool(v_post),
                    one_shot=bool(use_one_shot),
                    post_cot_gold_rationale=bool(v_gold_rationale),
                    post_cot_use_ig=bool(post_cot_use_ig),
                    post_cot_ig_steps=int(post_cot_ig_steps or 8),
                    post_cot_ig_top_frac=float(post_cot_ig_top_frac or 0.3),
                    filter_by_gold_answer=bool(v_filter),
                )

                cot_key = _cot_variant_key(
                    post=bool(v_post),
                    gran=int(granularity_level or 0),
                    one_shot=bool(use_one_shot),
                    multi_level=bool(granularity_multi_level),
                    gold_rationale=bool(v_gold_rationale),
                    ig=bool(post_cot_use_ig),
                    filter_gold=bool(v_filter),
                )
                tcf[cot_key] = str(cot_file)
                if cot_file_for_legacy is None:
                    cot_file_for_legacy = str(cot_file)

            state["teacher_cot_files"] = dict(tcf)
            # Legacy key (kept for backward-compat)
            if cot_file_for_legacy is not None:
                state["teacher_cot_file"] = str(cot_file_for_legacy)

        # Scientific validity fix: logits-cache is per KD mode/input. We only
        # build caches when logits-KD is enabled and tokenizers are compatible.
        if cache_logits and use_logits_kd:
            # Traditional mode cache (prompt+completion inputs)
            logits_dir_trad = cache_teacher_logits(
                teacher_model,
                teacher_tok,
                train_ds_for_cache,
                cache_root=cfg.cache_dir,
                batch_size=cfg.kd_params["batch_size"],
                device=str(cfg.device),
                generation_cfg=GenerationConfig(max_new_tokens=0, temperature=0.0, do_sample=False),
                split="train",
                seed=cfg.seeds[0],
                kd_mode="traditional",
                input_kind="prompt_completion",
                train_limit=cfg.train_limit,
                max_length=cfg.max_length,
            )
            state["teacher_logits_traditional_dir"] = str(logits_dir_trad)

            # Reasoning-aware mode cache requires teacher CoT first.
            if "reasoning" in kd_modes:
                if not state.get("teacher_cot_file"):
                    print(" (info) cache_logits para reasoning exige cache_cot; pulando logits do reasoning.")
                else:
                    from distill import build_reasoning_full_sequences_from_cot

                    use_one_shot = bool(granularity_one_shot) and int(granularity_level or 0) > 0
                    reason_key = _cot_variant_key(
                        post=bool(post_cot),
                        gran=int(granularity_level or 0),
                        one_shot=bool(use_one_shot),
                        multi_level=bool(granularity_multi_level),
                        gold_rationale=(bool(post_cot) and bool(post_cot_gold_rationale)),
                        ig=bool(post_cot_use_ig),
                        filter_gold=False,
                    )
                    cot_for_reasoning = None
                    tcf = state.get("teacher_cot_files")
                    if isinstance(tcf, dict):
                        cot_for_reasoning = tcf.get(reason_key)
                    cot_for_reasoning = cot_for_reasoning or state.get("teacher_cot_file")

                    full_sequences = build_reasoning_full_sequences_from_cot(
                        cot_path=str(cot_for_reasoning),
                        max_records=cfg.train_limit,
                        post_cot=bool(post_cot),
                        post_cot_use_ig=bool(post_cot_use_ig),
                    )
                    logits_dir_reason = cache_teacher_logits(
                        teacher_model,
                        teacher_tok,
                        full_sequences,
                        cache_root=cfg.cache_dir,
                        batch_size=cfg.kd_params["batch_size"],
                        device=str(cfg.device),
                        generation_cfg=GenerationConfig(max_new_tokens=0, temperature=0.0, do_sample=False),
                        split="train",
                        seed=cfg.seeds[0],
                        kd_mode="reasoning",
                        input_kind="prompt_completion",
                        train_limit=cfg.train_limit,
                        max_length=cfg.max_length,
                    )
                    state["teacher_logits_reasoning_dir"] = str(logits_dir_reason)

        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _save_state(state_path, state)

    teacher_logits_traditional_dir = state.get("teacher_logits_traditional_dir")
    teacher_logits_reasoning_dir = state.get("teacher_logits_reasoning_dir")

    # Backward-compat: older runs stored a single teacher_cot_file.
    teacher_cot_files = state.get("teacher_cot_files")
    if not isinstance(teacher_cot_files, dict):
        teacher_cot_files = {}
        legacy = state.get("teacher_cot_file")
        if legacy:
            # Assume legacy corresponds to current global settings (no filtering, no 1-shot).
            teacher_cot_files[
                _cot_variant_key(
                    post=bool(post_cot),
                    gran=int(granularity_level or 0),
                    one_shot=False,
                    multi_level=bool(granularity_multi_level),
                    gold_rationale=False,
                    ig=False,
                    filter_gold=False,
                )
            ] = str(legacy)
            state["teacher_cot_files"] = dict(teacher_cot_files)
            _save_state(state_path, state)

    for cond_name, cond_cfg in conditions.items():
        cond_runs = []

        # Record evaluation protocol per condition (independent of seed) so that
        # downstream comparisons don't silently mix protocols.
        cond_post_cot_preview = bool(post_cot) or bool(cond_cfg.get("force_post_cot", False))
        protocol_preview = {
            "use_cot_prompt": (False if bool(cond_post_cot_preview) else bool(use_cot_prompt_eval)),
            "answer_first_eval": bool(cond_post_cot_preview),
            "cascod_two_stage": bool(cond_cfg["mode"] == "cascod"),
        }
        prev = results.get("eval_protocols", {}).get(cond_name)
        if prev is None:
            results["eval_protocols"][cond_name] = dict(protocol_preview)
        elif prev != protocol_preview:
            raise ValueError(
                "Inconsistência interna: protocolo de avaliação variou dentro da mesma condição. "
                f"cond={cond_name} prev={prev} now={protocol_preview}"
            )

        for seed in cfg.seeds:
            run_key = f"{cond_name}_seed{seed}"
            if state.get("completed", {}).get(run_key):
                print(f" Pulando run já completado: {run_key}")
                cond_runs.append(state["completed"][run_key])
                continue

            print(f"\n Condição {cond_name} | seed={seed}")
            set_seed(seed)

            cond_post_cot = bool(post_cot) or bool(cond_cfg.get("force_post_cot", False))
            cond_one_shot = bool(granularity_one_shot) and int(granularity_level or 0) > 0
            cond_gold_rationale = bool(post_cot_gold_rationale) and bool(cond_post_cot)
            cond_filter_gold = bool(cascod_filter_by_gold) if cond_cfg["mode"] == "cascod" else False
            cot_key = _cot_variant_key(
                post=bool(cond_post_cot),
                gran=int(granularity_level or 0),
                one_shot=bool(cond_one_shot),
                multi_level=bool(granularity_multi_level),
                gold_rationale=bool(cond_gold_rationale),
                ig=bool(post_cot_use_ig),
                filter_gold=bool(cond_filter_gold),
            )
            teacher_cot_file = teacher_cot_files.get(cot_key)

            # Load models
            teacher_model = None
            teacher_tok = None
            if cond_cfg["mode"] in ("traditional", "cascod") and not teacher_logits_traditional_dir and use_logits_kd:
                teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)
            if cond_cfg["mode"] in ("reasoning", "cascod") and not teacher_cot_file:
                teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)
            if cond_cfg["mode"] == "reasoning" and use_logits_kd and not teacher_logits_reasoning_dir:
                # Only needed if logits-KD is requested for reasoning mode.
                teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)

            student_name = cfg.model_hierarchy[student_key]
            student_model, student_tok = setup_model_and_tokenizer(student_name, cfg.device, cfg.quantization)

            # Scientific validity fix: logits-KD requires tokenizer compatibility.
            # CasCoD fiel (q->r + q,r->a) não depende de logits-KD por padrão.
            if use_logits_kd and cond_cfg["mode"] != "cascod":
                cache_dir = None
                if cond_cfg["mode"] in ("traditional", "cascod"):
                    cache_dir = teacher_logits_traditional_dir
                elif cond_cfg["mode"] == "reasoning":
                    cache_dir = teacher_logits_reasoning_dir
                if cache_dir:
                    meta = read_cache_metadata(Path(cache_dir)) or {}
                    cache_hash = str(meta.get("tokenizer_hash") or "")
                    stud_hash = _tokenizer_compat_key(student_tok)
                    if cache_hash and stud_hash and cache_hash != stud_hash:
                        raise ValueError(
                            "Configuração cientificamente inválida: cache de logits foi gerado com tokenizer diferente do student. "
                            f"cache_tokenizer_hash='{cache_hash}', student_tokenizer_hash='{stud_hash}'."
                        )
                    # Backward-compat for older caches.
                    if not cache_hash:
                        cache_tok = str(meta.get("tokenizer") or "")
                        stud_name = str(getattr(student_tok, "name_or_path", ""))
                        if cache_tok and stud_name and cache_tok != stud_name:
                            raise ValueError(
                                "Configuração cientificamente inválida: cache antigo de logits foi gerado com tokenizer name_or_path diferente. "
                                f"cache_tokenizer='{cache_tok}', student_tokenizer='{stud_name}'."
                            )
                else:
                    if teacher_tok is None and cond_cfg["mode"] in ("traditional", "cascod"):
                        # Load teacher tokenizer just for compatibility check.
                        teacher_tok = AutoTokenizer.from_pretrained(cfg.model_hierarchy["teacher_medium"])
                    assert_tokenizer_compatible_for_logits_kd(
                        teacher_tok,
                        student_tok,
                        context=f"mode={cond_cfg['mode']}, student={student_key}",
                    )

            # Data
            train_ds = load_training_dataset(enable_gsm8k_train, enable_bbh_train, cfg.train_limit, seed=seed)

            # Scientific validity fix: reasoning/cascod conditions require teacher-generated CoT
            # (cached or generated on demand). No silent fallback to gold answers.
            if cond_cfg["mode"] in ("reasoning", "cascod") and not teacher_cot_file:
                if teacher_model is None or teacher_tok is None:
                    teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)
                teacher_cot_file = str(
                    cache_teacher_cot(
                        teacher_model,
                        teacher_tok,
                        train_ds,
                        cache_root=cfg.cache_dir,
                        batch_size=max(1, cfg.kd_params["batch_size"] // 2),
                        device=str(cfg.device),
                        generation_cfg=cfg.teacher_cot_generation,
                        split="train",
                        seed=seed,
                        prompt_max_length=cfg.max_length,
                        train_limit=cfg.train_limit,
                        granularity_level=int(granularity_level or 0),
                        granularity_multi_level=bool(granularity_multi_level),
                        post_cot=bool(cond_post_cot),
                        one_shot=bool(cond_one_shot),
                        post_cot_gold_rationale=bool(cond_gold_rationale),
                        post_cot_use_ig=bool(post_cot_use_ig),
                        post_cot_ig_steps=int(post_cot_ig_steps or 8),
                        post_cot_ig_top_frac=float(post_cot_ig_top_frac or 0.3),
                        filter_by_gold_answer=bool(cond_filter_gold),
                    )
                )
                teacher_cot_files[cot_key] = str(teacher_cot_file)
                state["teacher_cot_files"] = dict(teacher_cot_files)
                _save_state(state_path, state)

            # Note: CasCoD fiel (q->r + q,r->a) não precisa de logits cache.

            # Distill
            if cond_cfg["mode"] == "traditional":
                if not use_logits_kd:
                    raise ValueError(
                        "KD tradicional (logits-based) exige logits-KD habilitado. "
                        "Desabilitar logits-KD tornaria esta condição um SFT answer-only, "
                        "o que foge do desenho experimental atual."
                    )

                # Optional: save intermediate checkpoints per epoch for debugging.
                if str(os.environ.get("SLM_ENABLE_KD_EPOCH_CKPTS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}:
                    ckpt_dir = cfg.models_dir / exp_id / f"{cond_name}_seed{seed}_ckpts"
                    os.environ["SLM_KD_CKPT_DIR"] = str(ckpt_dir)
                    os.environ["SLM_KD_SAVE_EVERY_EPOCH"] = "1"

                distiller = TraditionalKDDistiller(cfg, cache_dir=teacher_logits_traditional_dir)
                trained_model, train_metrics = distiller.distill(
                    student_model,
                    teacher_model,
                    student_tok,
                    train_ds,
                    seed=seed,
                    use_cache=bool(teacher_logits_traditional_dir),
                )
            elif cond_cfg["mode"] == "reasoning":
                # For incompatible teacher/student, logits-KD is disabled and we fall back
                # to CE-only on distilled sequences (sequence-level distillation).
                distiller = ReasoningAwareDistiller(
                    cfg,
                    cot_cache_path=teacher_cot_file,
                    logits_cache_dir=(teacher_logits_reasoning_dir if use_logits_kd else None),
                )
                trained_model, train_metrics = distiller.distill_with_reasoning(
                    student_model,
                    teacher_model,
                    student_tok,
                    train_ds,
                    seed=seed,
                    use_cot_cache=bool(teacher_cot_file),
                    use_logits_cache=bool(use_logits_kd and teacher_logits_reasoning_dir),
                    granularity_level=int(granularity_level or 0),
                    post_cot=bool(cond_post_cot),
                    granularity_multi_level=bool(granularity_multi_level),
                    post_cot_use_ig=bool(post_cot_use_ig),
                )
            elif cond_cfg["mode"] == "cascod":
                if not teacher_cot_file:
                    raise ValueError("CasCoD exige teacher_cot_file.")

                # Faithful CasCoD uses Pre-CoT internally (q->r), and q,r->a for the answer.
                if bool(cond_post_cot):
                    print(" (info) CasCoD fiel usa Pre-CoT internamente; ignorando post_cot para o treino CasCoD.")

                import json as _json

                try:
                    from datasets import Dataset as _Dataset
                except Exception:
                    _Dataset = None

                records_r = []
                records_a = []
                total_lines = 0
                skipped_no_q = 0
                skipped_no_r = 0
                skipped_no_a = 0
                skipped_bad_json = 0
                with open(str(teacher_cot_file), "r", encoding="utf-8") as handle:
                    for line in handle:
                        total_lines += 1
                        try:
                            rec = _json.loads(line)
                        except Exception:
                            skipped_bad_json += 1
                            continue

                        q = (rec.get("question") or rec.get("text") or "").strip()
                        r = (rec.get("teacher_reasoning") or "").strip()
                        gold = (rec.get("gold_answer") or "").strip()
                        a = (gold or (rec.get("teacher_answer") or "")).strip()

                        if not q:
                            skipped_no_q += 1
                            continue
                        if not r:
                            skipped_no_r += 1
                            continue
                        if not a:
                            skipped_no_a += 1
                            continue

                        prompt_r = build_cascod_rationale_prompt(str(q), granularity_level=int(granularity_level or 0))
                        prompt_a = build_cascod_answer_prompt(str(q), str(r))

                        records_r.append({"text": prompt_r + r, "prompt": prompt_r, "completion": r})
                        records_a.append({"text": prompt_a + a, "prompt": prompt_a, "completion": a})

                if not records_r or not records_a:
                    raise ValueError(
                        "CasCoD: datasets vazios após processar o teacher_cot_file. "
                        f"total_lines={total_lines} bad_json={skipped_bad_json} "
                        f"skipped_no_q={skipped_no_q} skipped_no_r={skipped_no_r} skipped_no_a={skipped_no_a}. "
                        "Dica: verifique se o cache de CoT do teacher contém 'teacher_reasoning' e 'teacher_answer' (ou 'gold_answer'), "
                        "e se filtros como --cascod_filter_by_gold não estão eliminando quase tudo."
                    )

                if _Dataset is None:
                    raise RuntimeError("CasCoD requer o pacote 'datasets' instalado para construir datasets derivados.")

                ds_r = _Dataset.from_list(records_r)
                ds_a = _Dataset.from_list(records_a)

                old_epochs = int(cfg.kd_params.get("epochs", 1))
                stage1_epochs = max(1, int(cascod_stage1_epochs or 1))
                stage2_epochs = int(cascod_stage2_epochs or -1)
                if stage2_epochs <= 0:
                    stage2_epochs = old_epochs

                # Stage 1 warmup: q -> r
                cfg.kd_params["epochs"] = stage1_epochs
                model_s1, metrics_s1 = _train_sft_lora(cfg, student_model, student_tok, ds_r, seed=seed)

                stage1_dir = cfg.models_dir / exp_id / f"{cond_name}_stage1_seed{seed}"
                stage1_dir.mkdir(parents=True, exist_ok=True)
                model_s1.save_pretrained(stage1_dir)
                student_tok.save_pretrained(stage1_dir)

                # Stage 2: mixed loss L=(1-α)Lr + αLa
                cfg.kd_params["epochs"] = max(1, stage2_epochs)
                model_s2, metrics_s2 = _train_cascod_lora(
                    cfg,
                    model_s1,
                    student_tok,
                    ds_r,
                    ds_a,
                    seed=seed,
                    alpha=float(cascod_alpha),
                )

                cfg.kd_params["epochs"] = old_epochs

                trained_model = model_s2
                train_metrics = {
                    "cascod": {
                        "alpha": float(cascod_alpha),
                        "stage1": {
                            "kind": "q_to_r_sft",
                            "epochs": int(stage1_epochs),
                            "metrics": metrics_s1,
                            "ckpt_dir": str(stage1_dir),
                        },
                        "stage2": {
                            "kind": "mixed_loss_q_to_r_and_qr_to_a",
                            "epochs": int(stage2_epochs),
                            "metrics": metrics_s2,
                        },
                    }
                }
            else:
                raise ValueError(f"Modo desconhecido: {cond_cfg['mode']}")

            # Save model immediately after training (before potentially long eval).
            save_dir = cfg.models_dir / exp_id / f"{cond_name}_seed{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)
            trained_model.save_pretrained(save_dir)
            student_tok.save_pretrained(save_dir)
            print(f" Modelo salvo em: {save_dir}")

            # Eval
            eval_protocol = {
                "use_cot_prompt": (False if bool(cond_post_cot) else bool(use_cot_prompt_eval)),
                "answer_first_eval": bool(cond_post_cot),
                "cascod_two_stage": bool(cond_cfg["mode"] == "cascod"),
            }
            eval_results = evaluator.evaluate(
                trained_model,
                student_tok,
                seed=seed,
                eval_gsm8k=eval_gsm8k,
                eval_bbh=eval_bbh,
                eval_obqa=bool(eval_obqa),
                eval_efficiency=eval_efficiency,
                use_cot_prompt=bool(eval_protocol["use_cot_prompt"]),
                answer_first_eval=bool(eval_protocol["answer_first_eval"]),
                cascod_two_stage=bool(eval_protocol["cascod_two_stage"]),
                generation_cfg=cfg.eval_generation,
            )

            run_payload = {
                "seed": seed,
                "condition": cond_name,
                "description": cond_cfg["description"],
                "training": train_metrics,
                "evaluation_protocol": dict(eval_protocol),
                "evaluation": eval_results,
            }

            if cond_cfg["mode"] == "cascod":
                run_payload.setdefault("artifacts", {})
                run_payload["artifacts"].update(
                    {
                        "teacher_cot_file": str(teacher_cot_file),
                        "cascod_alpha": float(cascod_alpha),
                    }
                )

            state.setdefault("completed", {})[run_key] = run_payload
            _save_state(state_path, state)
            cond_runs.append(run_payload)

            del trained_model
            if teacher_model is not None:
                del teacher_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["conditions"][cond_name] = {"description": cond_cfg["description"], "runs": cond_runs}

    # Global protocol comparability check (warn by default; strict can abort).
    try:
        proto_items = list((results.get("eval_protocols") or {}).items())
        uniq = {}
        for k, v in proto_items:
            sig = json.dumps(v, sort_keys=True, default=str)
            uniq.setdefault(sig, []).append(k)
        if len(uniq) > 1:
            msg = "Múltiplos protocolos de avaliação detectados entre condições: " + "; ".join(
                [f"conds={names} proto={json.loads(sig)}" for sig, names in uniq.items()]
            )
            results.setdefault("warnings", []).append(msg)
            if bool(eval_protocol_strict):
                raise ValueError(msg)
            print(f"[WARN] {msg}")
    except Exception:
        pass

    # Hypothesis testing: compare metric across runs (paired by seed)
    if "kd_traditional" in results["conditions"] and "kd_with_reasoning" in results["conditions"]:
        p_a = (results.get("eval_protocols") or {}).get("kd_traditional")
        p_b = (results.get("eval_protocols") or {}).get("kd_with_reasoning")
        if p_a != p_b:
            msg = (
                "Protocolos de avaliação diferem entre condições; comparação estatística pode ser inválida. "
                f"kd_traditional={p_a}, kd_with_reasoning={p_b}."
            )
            if bool(eval_protocol_strict):
                raise ValueError(msg + " Defina SLM_EVAL_PROTOCOL_STRICT=0 para apenas avisar.")
            print(f"[WARN] {msg} Pulando hypothesis_testing.")
        else:
            trad = results["conditions"]["kd_traditional"]["runs"]
            reas = results["conditions"]["kd_with_reasoning"]["runs"]

            def _by_seed(runs: List[Dict[str, Any]]):
                m = {}
                for r in runs:
                    ev = r.get("evaluation", {})
                    m[int(r.get("seed"))] = float(ev.get(hypothesis_metric, ev.get("primary_score", ev.get("overall_score", 0.0))))
                return m

            a = _by_seed(trad)
            b = _by_seed(reas)
            common = sorted(set(a.keys()) & set(b.keys()))
            metrics_a = [a[s] for s in common]
            metrics_b = [b[s] for s in common]

            if len(common) < 2 and not allow_insufficient_runs:
                raise ValueError(
                    "Validade estatística insuficiente: n_runs < 2. "
                    "Adicione mais seeds (ex.: --seed 42 --seed 43) ou passe --allow_insufficient_runs para apenas registrar resultados sem inferência."  # noqa: E501
                )

            ht = analyst.paired_bootstrap_over_runs(metrics_a, metrics_b, n_bootstrap=cfg.bootstrap_samples, rng_seed=42)
            results["hypothesis_testing"] = {
                "h1": {
                    "statement": cfg.scientific_hypotheses.get("main"),
                    "metric": hypothesis_metric,
                    "n_runs": len(common),
                    "traditional_mean": float(sum(metrics_a) / len(metrics_a)) if metrics_a else None,
                    "reasoning_mean": float(sum(metrics_b) / len(metrics_b)) if metrics_b else None,
                    "test": ht,
                }
            }

    # Reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_json = cfg.reports_dir / f"comprehensive_report_{exp_id}_{timestamp}.json"
    summary_txt = cfg.reports_dir / f"results_summary_{exp_id}_{timestamp}.txt"

    results.setdefault("artifacts", {})
    results["artifacts"].update(
        {
            "experiment_id": exp_id,
            "experiment_dir": str(exp_dir),
            "report_json": str(report_json),
            "summary_txt": str(summary_txt),
        }
    )

    write_report_json(report_json, results)
    write_summary_txt(summary_txt, results)

    plot_paths = write_plots(cfg.reports_dir, results, prefix=f"plots_{exp_id}_{timestamp}")
    results["artifacts"]["plots"] = [str(p) for p in plot_paths]
    # Update JSON to include plot artifacts.
    write_report_json(report_json, results)

    print(" Relatórios salvos:")
    print(f"   - JSON: {report_json}")
    print(f"   - TXT:  {summary_txt}")
    if plot_paths:
        print("   - PLOTS:")
        for p in plot_paths:
            print(f"     * {p}")

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="H1 experiment: KD traditional vs CoT-aware KD")
    p.add_argument(
        "--drive_root",
        type=str,
        default=None,
        help="Output root directory. In Colab, use /content/drive/MyDrive/SLM_results (requires drive.mount).",
    )
    p.add_argument(
        "--kd_modes",
        nargs="+",
        default=["traditional", "reasoning"],
        choices=["traditional", "reasoning", "cascod", "combo_d"],
    )
    p.add_argument("--student", default="student_primary", choices=["student_primary", "student_small"])

    p.add_argument("--seed", type=int, action="append", dest="seeds", help="Repeatable. Example: --seed 42 --seed 43")
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--max_length", type=int, default=512)

    p.add_argument("--batch_size", type=int, default=None, help="Micro-batch size (default from config)")
    p.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps (default from config)")
    p.add_argument("--epochs", type=int, default=None, help="Training epochs (default from config)")

    p.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit loading (QLoRA-style when combined with LoRA)")
    p.add_argument("--no_load_in_4bit", action="store_true", help="Disable 4-bit loading")

    # Scientific control fix: explicit enable/disable flags; no default=True store_true.
    p.add_argument("--enable_gsm8k_train", action="store_true", help="Use GSM8K train split for training")
    p.add_argument("--disable_gsm8k_train", action="store_true", help="Disable GSM8K training")
    p.add_argument("--enable_bbh_train", action="store_true", help="Enable BBH/BBEH-derived training split (non-overlapping with eval)")
    p.add_argument("--disable_bbh_train", action="store_true", help="Disable BBH training")

    p.add_argument("--eval_gsm8k", action="store_true", help="Evaluate on GSM8K test")
    p.add_argument("--no_eval_gsm8k", action="store_true", help="Disable GSM8K evaluation")
    p.add_argument("--eval_bbh", action="store_true", help="Evaluate on BBEH/BBH held-out split")
    p.add_argument("--no_eval_bbh", action="store_true", help="Disable BBH evaluation")

    # Commonsense OOD (eval-only)
    p.add_argument("--eval_obqa", action="store_true", help="Evaluate on OpenBookQA test (commonsense; OOD; eval-only)")
    p.add_argument("--no_eval_obqa", action="store_true", help="Disable OBQA evaluation")
    p.add_argument("--eval_limit_obqa", type=int, default=200, help="Max OBQA examples to evaluate")
    p.add_argument("--eval_efficiency", action="store_true", help="Compute secondary efficiency metrics")
    p.add_argument("--no_eval_efficiency", action="store_true", help="Disable efficiency metrics")

    p.add_argument("--use_cot_prompt_eval", action="store_true", help="Use CoT-style prompt during evaluation")
    p.add_argument("--no_cot_prompt_eval", action="store_true", help="Use direct-answer prompt during evaluation")

    p.add_argument("--prepare_caches", action="store_true", help="Precompute caches (teacher CoT/logits) for faster reruns")
    p.add_argument("--cache_logits", action="store_true", help="Enable logits cache preparation (requires --use_logits_kd)")
    p.add_argument("--cache_cot", action="store_true", help="Enable teacher CoT cache preparation")

    p.add_argument("--use_logits_kd", action="store_true", help="Enable logits-based KD loss (requires tokenizer compatibility)")
    p.add_argument("--no_logits_kd", action="store_true", help="Disable logits-based KD (reasoning mode will fall back to CE-only)")

    p.add_argument(
        "--allow_insufficient_runs",
        action="store_true",
        help="Allow n_runs<2 (no valid hypothesis test); still writes reports.",
    )

    # Scientific QA / reproducibility controls (map to env vars consumed across the repo).
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (sets SLM_DETERMINISTIC=1)")
    p.add_argument("--no_deterministic", action="store_true", help="Disable deterministic mode (sets SLM_DETERMINISTIC=0)")

    p.add_argument(
        "--eval_protocol_strict",
        action="store_true",
        help="Abort when comparing conditions with different eval protocols (sets SLM_EVAL_PROTOCOL_STRICT=1)",
    )
    p.add_argument(
        "--no_eval_protocol_strict",
        action="store_true",
        help="Warn-only on eval protocol mismatch (sets SLM_EVAL_PROTOCOL_STRICT=0)",
    )

    # Reasoning mask quality gating (ReasoningAwareDistiller).
    p.add_argument("--reasoning_mask_strict", action="store_true", help="Abort on low-quality reasoning mask (SLM_REASONING_MASK_STRICT=1)")
    p.add_argument("--no_reasoning_mask_strict", action="store_true", help="Warn-only on low-quality reasoning mask (SLM_REASONING_MASK_STRICT=0)")
    p.add_argument(
        "--reasoning_mask_fallback_to_completion",
        action="store_true",
        help="If markers are missing/empty mask, fallback KD mask to completion (SLM_REASONING_MASK_FALLBACK_TO_COMPLETION=1)",
    )
    p.add_argument(
        "--no_reasoning_mask_fallback_to_completion",
        action="store_true",
        help="Disable fallback-to-completion for KD mask (SLM_REASONING_MASK_FALLBACK_TO_COMPLETION=0)",
    )
    p.add_argument(
        "--reasoning_mask_max_fallback",
        type=float,
        default=None,
        help="Max allowed fallback_rate before warning/abort (SLM_REASONING_MASK_MAX_FALLBACK)",
    )
    p.add_argument(
        "--reasoning_mask_min_reasoning_frac",
        type=float,
        default=None,
        help="Min allowed reasoning_token_fraction_mean before warning/abort (SLM_REASONING_MASK_MIN_REASONING_FRAC)",
    )

    # Logits cache loading (avoid OOM by streaming shards).
    p.add_argument(
        "--logits_cache_load_mode",
        choices=["auto", "eager", "lazy"],
        default=None,
        help="How to load logits cache: auto/eager/lazy (SLM_LOGITS_CACHE_LOAD_MODE)",
    )
    p.add_argument(
        "--logits_cache_eager_max_mb",
        type=float,
        default=None,
        help="When load_mode=auto, max estimated MB for eager concatenation (SLM_LOGITS_CACHE_EAGER_MAX_MB)",
    )
    p.add_argument(
        "--logits_cache_require_exact_match",
        action="store_true",
        help="Abort if logits cache length != built examples (SLM_LOGITS_CACHE_REQUIRE_EXACT_MATCH=1)",
    )
    p.add_argument(
        "--no_logits_cache_require_exact_match",
        action="store_true",
        help="Warn and auto-disable cache on mismatch (SLM_LOGITS_CACHE_REQUIRE_EXACT_MATCH=0)",
    )
    p.add_argument("--cot_cache_strict", action="store_true", help="Abort on CoT cache integrity warnings (SLM_COT_CACHE_STRICT=1)")
    p.add_argument("--no_cot_cache_strict", action="store_true", help="Warn-only on CoT cache issues (SLM_COT_CACHE_STRICT=0)")
    p.add_argument(
        "--logits_cache_allow_shuffle",
        action="store_true",
        help="Allow full random shuffle with lazy logits cache (may increase IO) (SLM_LOGITS_CACHE_ALLOW_SHUFFLE=1)",
    )
    p.add_argument(
        "--no_logits_cache_allow_shuffle",
        action="store_true",
        help="Use shard-aware shuffle for lazy cache (default) (SLM_LOGITS_CACHE_ALLOW_SHUFFLE=0)",
    )

    # Numeric stability policy (sanitization/clamp).
    p.add_argument("--train_sanitize_logits", action="store_true", help="Enable training logit sanitization (SLM_TRAIN_SANITIZE_LOGITS=1)")
    p.add_argument("--no_train_sanitize_logits", action="store_true", help="Disable training logit sanitization (SLM_TRAIN_SANITIZE_LOGITS=0)")
    p.add_argument(
        "--train_max_logit_abs",
        type=float,
        default=None,
        help="Clamp abs(logit) during training sanitization (SLM_TRAIN_MAX_LOGIT_ABS)",
    )

    p.add_argument("--cache_sanitize_logits", action="store_true", help="Enable cache sanitization before fp16 cast (SLM_CACHE_SANITIZE_LOGITS=1)")
    p.add_argument("--no_cache_sanitize_logits", action="store_true", help="Disable cache sanitization (SLM_CACHE_SANITIZE_LOGITS=0)")
    p.add_argument("--cache_clamp_fp16", action="store_true", help="Clamp logits to fp16-safe range when caching (SLM_CACHE_CLAMP_FP16=1)")
    p.add_argument("--no_cache_clamp_fp16", action="store_true", help="Disable fp16 clamp when caching (SLM_CACHE_CLAMP_FP16=0)")
    p.add_argument(
        "--cache_fp16_safe_abs",
        type=float,
        default=None,
        help="Max abs(logit) allowed in fp16 cache (SLM_CACHE_FP16_SAFE_ABS)",
    )

    p.add_argument(
        "--hypothesis_metric",
        default="primary_score",
        choices=["primary_score", "overall_score"],
        help="Metric used for H1 statistical test (default: primary_score).",
    )

    # Technique 2B: CasCoD (two-stage cascade distillation).
    p.add_argument("--cascod_stage1_epochs", type=int, default=1, help="CasCoD stage1 epochs (reasoning CE-only)")
    p.add_argument(
        "--cascod_stage2_epochs",
        type=int,
        default=-1,
        help="CasCoD stage2 epochs (traditional logits-KD). -1 means use --epochs",
    )

    p.add_argument(
        "--cascod_alpha",
        type=float,
        default=0.3,
        help="CasCoD alpha in L=(1-alpha)L_rationale + alpha L_answer (default: 0.3)",
    )

    p.add_argument("--eval_max_new_tokens", type=int, default=256)
    p.add_argument("--eval_temperature", type=float, default=0.0)

    p.add_argument("--teacher_cot_max_new_tokens", type=int, default=150)
    p.add_argument("--teacher_cot_temperature", type=float, default=0.0)

    # Technique 2A: reasoning granularity adaptation.
    p.add_argument(
        "--granularity_level",
        type=int,
        default=0,
        help="0=disabled (backward-compatible), 1..6 increasing reasoning detail for teacher CoT prompts",
    )

    p.add_argument("--granularity_one_shot", dest="granularity_one_shot", action="store_true", help="Enable 1-shot exemplar for granularity prompts")
    p.add_argument("--no_granularity_one_shot", dest="granularity_one_shot", action="store_false", help="Disable 1-shot exemplar for granularity prompts")
    p.set_defaults(granularity_one_shot=None)

    p.add_argument(
        "--granularity_multi_level",
        dest="granularity_multi_level",
        action="store_true",
        help="When granularity is enabled, cache all levels 1..level and expand training across levels",
    )
    p.add_argument(
        "--no_granularity_multi_level",
        dest="granularity_multi_level",
        action="store_false",
        help="Disable multi-level granularity caching/expansion (legacy: only cache the max level)",
    )
    p.set_defaults(granularity_multi_level=None)

    # Technique 2C: Post-CoT (answer-first training + eval without rationale).
    p.add_argument("--post_cot", dest="post_cot", action="store_true", help="Enable Post-CoT (answer-first) formatting")
    p.add_argument("--no_post_cot", dest="post_cot", action="store_false", help="Disable Post-CoT")
    p.set_defaults(post_cot=False)

    p.add_argument(
        "--post_cot_gold_rationale",
        dest="post_cot_gold_rationale",
        action="store_true",
        help="Post-CoT teacher prompt: provide gold answer and elicit justification only (more faithful to paper)",
    )
    p.add_argument(
        "--no_post_cot_gold_rationale",
        dest="post_cot_gold_rationale",
        action="store_false",
        help="Disable gold-conditioned rationale elicitation for Post-CoT (legacy behavior)",
    )
    p.set_defaults(post_cot_gold_rationale=None)

    p.add_argument(
        "--post_cot_use_ig",
        dest="post_cot_use_ig",
        action="store_true",
        help="Post-CoT: apply Integrated Gradients to filter/keep important rationale tokens (requires captum)",
    )
    p.add_argument(
        "--no_post_cot_use_ig",
        dest="post_cot_use_ig",
        action="store_false",
        help="Disable Integrated Gradients filtering for Post-CoT",
    )
    p.set_defaults(post_cot_use_ig=None)

    p.add_argument(
        "--post_cot_ig_steps",
        type=int,
        default=8,
        help="Integrated Gradients steps (only used when --post_cot_use_ig; default: 8)",
    )
    p.add_argument(
        "--post_cot_ig_top_frac",
        type=float,
        default=0.3,
        help="Fraction of rationale tokens to keep by IG importance (0,1]; default: 0.3)",
    )

    p.add_argument(
        "--cascod_filter_by_gold",
        dest="cascod_filter_by_gold",
        action="store_true",
        help="CasCoD stage1: filter teacher CoTs to keep only examples where teacher answer matches gold",
    )
    p.add_argument(
        "--no_cascod_filter_by_gold",
        dest="cascod_filter_by_gold",
        action="store_false",
        help="Disable CasCoD gold-match filtering (keeps all teacher CoTs)",
    )
    p.set_defaults(cascod_filter_by_gold=True)

    # Baseline 0.1
    p.add_argument(
        "--ft_teacher",
        action="store_true",
        help="Run baseline 0.1: supervised fine-tuning (SFT) of the teacher on TRAIN split only (LoRA), then evaluate and save checkpoint.",
    )
    p.add_argument(
        "--ft_student",
        action="store_true",
        help="Run baseline 0.2: supervised fine-tuning (SFT) of the student (no KD) on TRAIN split only (LoRA), then evaluate and save checkpoint.",
    )

    # Baseline 0.3
    p.add_argument(
        "--kd_logits_baseline",
        action="store_true",
        help="Run baseline 0.3: KD by logits using an adjusted teacher checkpoint dir (teacher_ckpt_dir) -> student.",
    )
    p.add_argument(
        "--teacher_ckpt_dir",
        type=str,
        default=None,
        help="Path to a saved teacher directory (either full HF model dir with config.json or PEFT adapter dir with adapter_config.json). Used by --kd_logits_baseline and --kd_cot_baseline.",
    )

    # Baseline 0.4
    p.add_argument(
        "--kd_cot_baseline",
        action="store_true",
        help="Run baseline 0.4: KD CoT padrão (text-only; CoT before answer; CE-only; no granularidade/CasCoD/Post-CoT).",
    )

    return p


def main(argv: Optional[Sequence[str]] = None):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_arg_parser().parse_args(argv)
    cfg = EvidenceBasedConfig(drive_root=Path(args.drive_root)) if args.drive_root else EvidenceBasedConfig()

    # Map CLI flags -> env vars used across modules.
    if bool(getattr(args, "deterministic", False)):
        os.environ["SLM_DETERMINISTIC"] = "1"
    if bool(getattr(args, "no_deterministic", False)):
        os.environ["SLM_DETERMINISTIC"] = "0"
    if bool(getattr(args, "eval_protocol_strict", False)):
        os.environ["SLM_EVAL_PROTOCOL_STRICT"] = "1"
    if bool(getattr(args, "no_eval_protocol_strict", False)):
        os.environ["SLM_EVAL_PROTOCOL_STRICT"] = "0"

    if bool(getattr(args, "reasoning_mask_strict", False)):
        os.environ["SLM_REASONING_MASK_STRICT"] = "1"
    if bool(getattr(args, "no_reasoning_mask_strict", False)):
        os.environ["SLM_REASONING_MASK_STRICT"] = "0"
    if bool(getattr(args, "reasoning_mask_fallback_to_completion", False)):
        os.environ["SLM_REASONING_MASK_FALLBACK_TO_COMPLETION"] = "1"
    if bool(getattr(args, "no_reasoning_mask_fallback_to_completion", False)):
        os.environ["SLM_REASONING_MASK_FALLBACK_TO_COMPLETION"] = "0"
    if getattr(args, "reasoning_mask_max_fallback", None) is not None:
        os.environ["SLM_REASONING_MASK_MAX_FALLBACK"] = str(float(args.reasoning_mask_max_fallback))
    if getattr(args, "reasoning_mask_min_reasoning_frac", None) is not None:
        os.environ["SLM_REASONING_MASK_MIN_REASONING_FRAC"] = str(float(args.reasoning_mask_min_reasoning_frac))

    if getattr(args, "logits_cache_load_mode", None):
        os.environ["SLM_LOGITS_CACHE_LOAD_MODE"] = str(args.logits_cache_load_mode)
    if getattr(args, "logits_cache_eager_max_mb", None) is not None:
        os.environ["SLM_LOGITS_CACHE_EAGER_MAX_MB"] = str(float(args.logits_cache_eager_max_mb))
    if bool(getattr(args, "logits_cache_require_exact_match", False)):
        os.environ["SLM_LOGITS_CACHE_REQUIRE_EXACT_MATCH"] = "1"
    if bool(getattr(args, "no_logits_cache_require_exact_match", False)):
        os.environ["SLM_LOGITS_CACHE_REQUIRE_EXACT_MATCH"] = "0"
    if bool(getattr(args, "cot_cache_strict", False)):
        os.environ["SLM_COT_CACHE_STRICT"] = "1"
    if bool(getattr(args, "no_cot_cache_strict", False)):
        os.environ["SLM_COT_CACHE_STRICT"] = "0"
    if bool(getattr(args, "logits_cache_allow_shuffle", False)):
        os.environ["SLM_LOGITS_CACHE_ALLOW_SHUFFLE"] = "1"
    if bool(getattr(args, "no_logits_cache_allow_shuffle", False)):
        os.environ["SLM_LOGITS_CACHE_ALLOW_SHUFFLE"] = "0"

    if bool(getattr(args, "train_sanitize_logits", False)):
        os.environ["SLM_TRAIN_SANITIZE_LOGITS"] = "1"
    if bool(getattr(args, "no_train_sanitize_logits", False)):
        os.environ["SLM_TRAIN_SANITIZE_LOGITS"] = "0"
    if getattr(args, "train_max_logit_abs", None) is not None:
        os.environ["SLM_TRAIN_MAX_LOGIT_ABS"] = str(float(args.train_max_logit_abs))

    if bool(getattr(args, "cache_sanitize_logits", False)):
        os.environ["SLM_CACHE_SANITIZE_LOGITS"] = "1"
    if bool(getattr(args, "no_cache_sanitize_logits", False)):
        os.environ["SLM_CACHE_SANITIZE_LOGITS"] = "0"
    if bool(getattr(args, "cache_clamp_fp16", False)):
        os.environ["SLM_CACHE_CLAMP_FP16"] = "1"
    if bool(getattr(args, "no_cache_clamp_fp16", False)):
        os.environ["SLM_CACHE_CLAMP_FP16"] = "0"
    if getattr(args, "cache_fp16_safe_abs", None) is not None:
        os.environ["SLM_CACHE_FP16_SAFE_ABS"] = str(float(args.cache_fp16_safe_abs))

    if args.seeds:
        cfg.seeds = list(args.seeds)
    if args.train_limit is not None:
        cfg.train_limit = args.train_limit
    cfg.max_length = args.max_length

    if args.batch_size is not None:
        cfg.kd_params["batch_size"] = int(args.batch_size)
    if args.grad_accum_steps is not None:
        cfg.kd_params["grad_accum_steps"] = int(args.grad_accum_steps)
    if args.epochs is not None:
        cfg.kd_params["epochs"] = int(args.epochs)

    if args.load_in_4bit:
        cfg.quantization["load_in_4bit"] = True
    if args.no_load_in_4bit:
        cfg.quantization["load_in_4bit"] = False

    cfg.eval_generation = GenerationConfig(max_new_tokens=args.eval_max_new_tokens, temperature=args.eval_temperature, do_sample=False)
    cfg.teacher_cot_generation = GenerationConfig(
        max_new_tokens=args.teacher_cot_max_new_tokens, temperature=args.teacher_cot_temperature, do_sample=False
    )

    granularity_level = int(getattr(args, "granularity_level", 0) or 0)
    post_cot = bool(getattr(args, "post_cot", False))

    # Defaults that track the paper more closely:
    # - Granularity adaptation: include a 1-shot exemplar when granularity is enabled.
    # - Post-CoT: teacher justifications are elicited conditioned on the gold answer.
    go_raw = getattr(args, "granularity_one_shot", None)
    granularity_one_shot = bool(go_raw) if go_raw is not None else bool(granularity_level > 0)

    gm_raw = getattr(args, "granularity_multi_level", None)
    granularity_multi_level = bool(gm_raw) if gm_raw is not None else bool(granularity_level > 0)

    pr_raw = getattr(args, "post_cot_gold_rationale", None)
    post_cot_gold_rationale = bool(pr_raw) if pr_raw is not None else bool(post_cot)

    ig_raw = getattr(args, "post_cot_use_ig", None)
    # Default: keep IG off unless explicitly enabled (requires captum).
    post_cot_use_ig = bool(ig_raw) if ig_raw is not None else False
    post_cot_ig_steps = int(getattr(args, "post_cot_ig_steps", 8) or 8)
    post_cot_ig_top_frac = float(getattr(args, "post_cot_ig_top_frac", 0.3) or 0.3)

    if post_cot_ig_steps < 1:
        raise ValueError("--post_cot_ig_steps deve ser >= 1")
    if not (0.0 < post_cot_ig_top_frac <= 1.0):
        raise ValueError("--post_cot_ig_top_frac deve estar em (0, 1]")

    cascod_filter_by_gold = bool(getattr(args, "cascod_filter_by_gold", True))
    cascod_stage1_epochs = int(getattr(args, "cascod_stage1_epochs", 1) or 1)
    cascod_stage2_epochs = int(getattr(args, "cascod_stage2_epochs", -1) or -1)
    cascod_alpha = float(getattr(args, "cascod_alpha", 0.3) or 0.3)

    enable_gsm8k_train = True
    if args.enable_gsm8k_train:
        enable_gsm8k_train = True
    if args.disable_gsm8k_train:
        enable_gsm8k_train = False

    enable_bbh_train = bool(args.enable_bbh_train)
    if args.disable_bbh_train:
        enable_bbh_train = False

    eval_gsm8k = True
    if args.eval_gsm8k:
        eval_gsm8k = True
    if args.no_eval_gsm8k:
        eval_gsm8k = False

    eval_bbh = True
    if args.eval_bbh:
        eval_bbh = True
    if args.no_eval_bbh:
        eval_bbh = False

    eval_obqa = False
    if args.eval_obqa:
        eval_obqa = True
    if args.no_eval_obqa:
        eval_obqa = False
    cfg.eval_limit_obqa = int(args.eval_limit_obqa)

    eval_eff = True
    if args.eval_efficiency:
        eval_eff = True
    if args.no_eval_efficiency:
        eval_eff = False

    use_cot_eval = True
    if args.use_cot_prompt_eval:
        use_cot_eval = True
    if args.no_cot_prompt_eval:
        use_cot_eval = False

    prepare_caches = bool(args.prepare_caches)
    cache_logits = bool(args.cache_logits)
    cache_cot = bool(args.cache_cot)

    use_logits_kd = True
    if args.use_logits_kd:
        use_logits_kd = True
    if args.no_logits_kd:
        use_logits_kd = False

    # Baseline 0.1: FT Teacher
    if bool(args.ft_teacher):
        return run_ft_teacher_baseline(
            cfg,
            enable_gsm8k_train=enable_gsm8k_train,
            enable_bbh_train=enable_bbh_train,
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_obqa=eval_obqa,
            eval_efficiency=eval_eff,
            use_cot_prompt_eval=use_cot_eval,
        )

    # Baseline 0.2: FT Student
    if bool(args.ft_student):
        return run_ft_student_baseline(
            cfg,
            student_key=str(args.student),
            enable_gsm8k_train=enable_gsm8k_train,
            enable_bbh_train=enable_bbh_train,
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_obqa=eval_obqa,
            eval_efficiency=eval_eff,
            use_cot_prompt_eval=use_cot_eval,
        )

    # Baseline 0.3: KD logits (teacher adjusted -> student)
    if bool(args.kd_logits_baseline):
        if not args.teacher_ckpt_dir:
            raise ValueError("--kd_logits_baseline requer --teacher_ckpt_dir")
        return run_kd_logits_baseline(
            cfg,
            teacher_ckpt_dir=str(args.teacher_ckpt_dir),
            student_key=str(args.student),
            enable_gsm8k_train=enable_gsm8k_train,
            enable_bbh_train=enable_bbh_train,
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_obqa=eval_obqa,
            eval_efficiency=eval_eff,
            use_cot_prompt_eval=use_cot_eval,
        )

    # Baseline 0.4: KD CoT padrão (text-only)
    if bool(args.kd_cot_baseline):
        return run_kd_cot_standard_baseline(
            cfg,
            teacher_ckpt_dir=(str(args.teacher_ckpt_dir) if args.teacher_ckpt_dir else None),
            student_key=str(args.student),
            enable_gsm8k_train=enable_gsm8k_train,
            enable_bbh_train=enable_bbh_train,
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_obqa=eval_obqa,
            eval_efficiency=eval_eff,
            use_cot_prompt_eval=use_cot_eval,
            # Baseline 0.4 is intentionally "standard" (no granularity/Post-CoT/CasCoD knobs).
            granularity_level=0,
        )

    return run_experiment(
        cfg,
        kd_modes=args.kd_modes,
        enable_gsm8k_train=enable_gsm8k_train,
        enable_bbh_train=enable_bbh_train,
        eval_gsm8k=eval_gsm8k,
        eval_bbh=eval_bbh,
        eval_obqa=eval_obqa,
        eval_efficiency=eval_eff,
        use_cot_prompt_eval=use_cot_eval,
        prepare_caches=prepare_caches,
        cache_logits=cache_logits,
        cache_cot=cache_cot,
        use_logits_kd=use_logits_kd,
        allow_insufficient_runs=bool(args.allow_insufficient_runs),
        hypothesis_metric=str(args.hypothesis_metric),
        student_key=str(args.student),
        granularity_level=granularity_level,
        granularity_one_shot=granularity_one_shot,
        granularity_multi_level=granularity_multi_level,
        cascod_stage1_epochs=cascod_stage1_epochs,
        cascod_stage2_epochs=cascod_stage2_epochs,
        cascod_alpha=cascod_alpha,
        post_cot=post_cot,
        post_cot_gold_rationale=post_cot_gold_rationale,
        post_cot_use_ig=post_cot_use_ig,
        post_cot_ig_steps=post_cot_ig_steps,
        post_cot_ig_top_frac=post_cot_ig_top_frac,
        cascod_filter_by_gold=cascod_filter_by_gold,
    )


if __name__ == "__main__":
    main()
