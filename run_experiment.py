from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cache import cache_teacher_cot, cache_teacher_logits, read_cache_metadata, tokenizer_fingerprint
from config import EvidenceBasedConfig, GenerationConfig, ensure_tokenizer_has_pad, get_safe_tokenizer_length, safe_model_to, set_seed
from data import load_training_dataset
from distill import ReasoningAwareDistiller, TraditionalKDDistiller, autocast_ctx, make_grad_scaler, preprocess_and_tokenize
from eval import StandardizedEvaluator
from report import ScientificLogger, write_plots, write_report_json, write_summary_txt
from stats import StatisticalAnalyst


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

            scaler.scale(loss_to_backprop).backward()

            if (step % grad_accum_steps) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            metrics["losses"].append(float(loss.detach().cpu().item()))
            if metrics["losses"]:
                recent = metrics["losses"][-min(50, len(metrics["losses"])) :]
                pbar.set_postfix({"loss": f"{sum(recent)/len(recent):.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

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
    cascod_stage1_epochs: int = 1,
    cascod_stage2_epochs: int = -1,
    post_cot: bool = False,
) -> Dict[str, Any]:
    logger = ScientificLogger()
    analyst = StatisticalAnalyst(alpha=cfg.alpha_level)
    evaluator = StandardizedEvaluator(cfg)

    combo_d_enabled = "combo_d" in set(kd_modes or [])

    def _cot_variant_key(*, post: bool, gran: int) -> str:
        return f"post_cot={int(bool(post))}|granularity={int(gran or 0)}"

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
        "cascod_stage1_epochs": int(cascod_stage1_epochs or 0),
        "cascod_stage2_epochs": int(cascod_stage2_epochs or -1),
        "post_cot": bool(post_cot),
        "combo_d_enabled": bool(combo_d_enabled),
    }

    exp_id = _experiment_id(cfg, kd_modes, flags)
    exp_dir = cfg.experiments_dir / exp_id
    state_path = exp_dir / "state.json"
    state = _load_state(state_path)

    logger.log_phase("EXPERIMENT", {"id": exp_id, "dir": str(exp_dir), "flags": flags})
    logger.log_hyperparameters(cfg.to_metadata())

    conditions: Dict[str, Dict[str, Any]] = {}
    if "traditional" in kd_modes:
        conditions["kd_traditional"] = {"mode": "traditional", "description": "KD tradicional (answer-only)"}
    if "reasoning" in kd_modes:
        conditions["kd_with_reasoning"] = {"mode": "reasoning", "description": "KD com raciocínio (CoT-aware)"}
    if "cascod" in kd_modes:
        conditions["kd_cascod"] = {"mode": "cascod", "description": "CasCoD (2B): 2 estágios (CoT CE-only → logits-KD)"}
    if "combo_d" in kd_modes:
        conditions["kd_combo_d"] = {
            "mode": "cascod",
            "description": "Condição D (A+B+C): granularity + CasCoD + Post-CoT",
            "force_post_cot": True,
        }
    if not conditions:
        raise ValueError("Nenhum modo de KD selecionado.")

    # Prepare datasets once per seed (training)
    results: Dict[str, Any] = {"metadata": cfg.to_metadata(), "conditions": {}}

    # Optional cache preparation
    if prepare_caches:
        # Prepare caches against the *training* split only (leakage control is
        # handled in data/eval modules via explicit splitting).
        train_ds_for_cache = load_training_dataset(enable_gsm8k_train, enable_bbh_train, cfg.train_limit, seed=cfg.seeds[0])

        teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)

        if cache_cot:
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
                post_cot=bool(post_cot),
            )
            # Legacy key (kept for backward-compat)
            state["teacher_cot_file"] = str(cot_file)

            # New variant-keyed mapping (prevents collisions across post_cot/granularity)
            cot_key = _cot_variant_key(post=bool(post_cot), gran=int(granularity_level or 0))
            tcf = state.get("teacher_cot_files")
            if not isinstance(tcf, dict):
                tcf = {}
            tcf[cot_key] = str(cot_file)
            state["teacher_cot_files"] = dict(tcf)

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

                    full_sequences = build_reasoning_full_sequences_from_cot(
                        cot_path=str(state["teacher_cot_file"]),
                        max_records=cfg.train_limit,
                        granularity_level=int(granularity_level or 0),
                        post_cot=bool(post_cot),
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
            # Assume legacy corresponds to current global settings.
            teacher_cot_files[_cot_variant_key(post=bool(post_cot), gran=int(granularity_level or 0))] = str(legacy)
            state["teacher_cot_files"] = dict(teacher_cot_files)
            _save_state(state_path, state)

    for cond_name, cond_cfg in conditions.items():
        cond_runs = []
        for seed in cfg.seeds:
            run_key = f"{cond_name}_seed{seed}"
            if state.get("completed", {}).get(run_key):
                print(f" Pulando run já completado: {run_key}")
                cond_runs.append(state["completed"][run_key])
                continue

            print(f"\n Condição {cond_name} | seed={seed}")
            set_seed(seed)

            cond_post_cot = bool(post_cot) or bool(cond_cfg.get("force_post_cot", False))
            cot_key = _cot_variant_key(post=cond_post_cot, gran=int(granularity_level or 0))
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
            if use_logits_kd:
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
                        post_cot=bool(cond_post_cot),
                    )
                )
                teacher_cot_files[cot_key] = str(teacher_cot_file)
                state["teacher_cot_files"] = dict(teacher_cot_files)
                _save_state(state_path, state)

            # CasCoD stage2 requires traditional logits cache when logits-KD is enabled.
            if cond_cfg["mode"] == "cascod" and use_logits_kd and not teacher_logits_traditional_dir:
                if teacher_model is None or teacher_tok is None:
                    teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)
                logits_dir_trad = cache_teacher_logits(
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
                teacher_logits_traditional_dir = str(logits_dir_trad)
                state["teacher_logits_traditional_dir"] = str(teacher_logits_traditional_dir)
                _save_state(state_path, state)

            # Distill
            if cond_cfg["mode"] == "traditional":
                if not use_logits_kd:
                    raise ValueError(
                        "KD tradicional (logits-based) exige logits-KD habilitado. "
                        "Desabilitar logits-KD tornaria esta condição um SFT answer-only, "
                        "o que foge do desenho experimental atual."
                    )
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
                )
            elif cond_cfg["mode"] == "cascod":
                if not use_logits_kd:
                    raise ValueError(
                        "CasCoD (2B) requer logits-KD habilitado para o estágio 2 (answer-only). "
                        "Use --use_logits_kd ou remova 'cascod' dos kd_modes."
                    )
                if not teacher_cot_file:
                    raise ValueError("CasCoD exige teacher_cot_file.")

                old_epochs = int(cfg.kd_params.get("epochs", 1))
                stage1_epochs = max(1, int(cascod_stage1_epochs or 1))
                stage2_epochs = int(cascod_stage2_epochs or -1)
                if stage2_epochs <= 0:
                    stage2_epochs = old_epochs

                # Stage 1: reasoning CE-only on teacher CoT.
                cfg.kd_params["epochs"] = stage1_epochs
                distiller1 = ReasoningAwareDistiller(cfg, cot_cache_path=teacher_cot_file, logits_cache_dir=None)
                model_s1, metrics_s1 = distiller1.distill_with_reasoning(
                    student_model,
                    teacher_model=None,
                    student_tokenizer=student_tok,
                    raw_dataset=train_ds,
                    seed=seed,
                    use_cot_cache=True,
                    use_logits_cache=False,
                    use_teacher_logits=False,
                    granularity_level=int(granularity_level or 0),
                    post_cot=bool(cond_post_cot),
                )

                stage1_dir = cfg.models_dir / exp_id / f"{cond_name}_stage1_seed{seed}"
                stage1_dir.mkdir(parents=True, exist_ok=True)
                model_s1.save_pretrained(stage1_dir)
                student_tok.save_pretrained(stage1_dir)

                # Stage 2: traditional logits-KD answer-only, continuing from stage1.
                cfg.kd_params["epochs"] = max(1, stage2_epochs)
                distiller2 = TraditionalKDDistiller(cfg, cache_dir=teacher_logits_traditional_dir)
                model_s2, metrics_s2 = distiller2.distill(
                    model_s1,
                    teacher_model,
                    student_tok,
                    train_ds,
                    seed=seed,
                    use_cache=bool(teacher_logits_traditional_dir),
                )

                cfg.kd_params["epochs"] = old_epochs

                trained_model = model_s2
                train_metrics = {
                    "cascod": {
                        "stage1": {
                            "kind": "reasoning_ce_only",
                            "epochs": int(stage1_epochs),
                            "metrics": metrics_s1,
                            "ckpt_dir": str(stage1_dir),
                            "granularity_level": int(granularity_level or 0),
                            "post_cot": bool(cond_post_cot),
                        },
                        "stage2": {
                            "kind": "traditional_logits_kd",
                            "epochs": int(stage2_epochs),
                            "metrics": metrics_s2,
                            "logits_cache_dir": str(teacher_logits_traditional_dir),
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
            eval_results = evaluator.evaluate(
                trained_model,
                student_tok,
                seed=seed,
                eval_gsm8k=eval_gsm8k,
                eval_bbh=eval_bbh,
                eval_obqa=bool(eval_obqa),
                eval_efficiency=eval_efficiency,
                use_cot_prompt=(False if bool(cond_post_cot) else use_cot_prompt_eval),
                answer_first_eval=bool(cond_post_cot),
                generation_cfg=cfg.eval_generation,
            )

            run_payload = {
                "seed": seed,
                "condition": cond_name,
                "description": cond_cfg["description"],
                "training": train_metrics,
                "evaluation": eval_results,
            }

            if cond_cfg["mode"] == "cascod":
                run_payload.setdefault("artifacts", {})
                run_payload["artifacts"].update(
                    {
                        "teacher_cot_file": str(teacher_cot_file),
                        "teacher_logits_traditional_dir": str(teacher_logits_traditional_dir),
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

    # Hypothesis testing: compare metric across runs (paired by seed)
    if "kd_traditional" in results["conditions"] and "kd_with_reasoning" in results["conditions"]:
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

    # Technique 2C: Post-CoT (answer-first training + eval without rationale).
    p.add_argument("--post_cot", dest="post_cot", action="store_true", help="Enable Post-CoT (answer-first) formatting")
    p.add_argument("--no_post_cot", dest="post_cot", action="store_false", help="Disable Post-CoT")
    p.set_defaults(post_cot=False)

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
    cascod_stage1_epochs = int(getattr(args, "cascod_stage1_epochs", 1) or 1)
    cascod_stage2_epochs = int(getattr(args, "cascod_stage2_epochs", -1) or -1)

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
            granularity_level=granularity_level,
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
        cascod_stage1_epochs=cascod_stage1_epochs,
        cascod_stage2_epochs=cascod_stage2_epochs,
        post_cot=post_cot,
    )


if __name__ == "__main__":
    main()
