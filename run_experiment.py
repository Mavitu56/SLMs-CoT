from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cache import cache_teacher_cot, cache_teacher_logits, read_cache_metadata
from config import EvidenceBasedConfig, GenerationConfig, ensure_tokenizer_has_pad, get_safe_tokenizer_length, safe_model_to, set_seed
from data import load_training_dataset
from distill import ReasoningAwareDistiller, TraditionalKDDistiller
from eval import StandardizedEvaluator
from report import ScientificLogger, write_report_json, write_summary_txt
from stats import StatisticalAnalyst


def _tokenizer_compat_key(tokenizer) -> str:
    """Tokenizer compatibility fingerprint.

    Scientific validity fix: logits-based KD assumes teacher forward-pass uses the
    exact same token IDs as the student.

    `name_or_path` differs across model sizes even when the tokenizer is
    identical (e.g., Qwen2.5 7B vs 3B), so we compare a stable hash of the full
    vocabulary mapping.
    """

    cached = getattr(tokenizer, "_slm_tokenizer_hash", None)
    if isinstance(cached, str) and cached:
        return cached

    import hashlib

    vocab = tokenizer.get_vocab()  # token -> id
    h = hashlib.sha256()
    h.update(tokenizer.__class__.__name__.encode("utf-8"))
    h.update(str(len(vocab)).encode("utf-8"))
    for k in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        h.update(f"|{k}={getattr(tokenizer, k, None)}".encode("utf-8"))
    for token, idx in sorted(vocab.items(), key=lambda kv: kv[0]):
        h.update(token.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(str(int(idx)).encode("utf-8"))
        h.update(b"\n")
    digest = h.hexdigest()[:16]
    try:
        setattr(tokenizer, "_slm_tokenizer_hash", digest)
    except Exception:
        pass
    return digest


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
    eval_efficiency: bool,
    use_cot_prompt_eval: bool,
    prepare_caches: bool,
    cache_logits: bool,
    cache_cot: bool,
    use_logits_kd: bool,
    allow_insufficient_runs: bool,
    hypothesis_metric: str,
    student_key: str,
) -> Dict[str, Any]:
    logger = ScientificLogger()
    analyst = StatisticalAnalyst(alpha=cfg.alpha_level)
    evaluator = StandardizedEvaluator(cfg)

    flags = {
        "enable_gsm8k_train": enable_gsm8k_train,
        "enable_bbh_train": enable_bbh_train,
        "eval_gsm8k": eval_gsm8k,
        "eval_bbh": eval_bbh,
        "eval_efficiency": eval_efficiency,
        "use_cot_prompt_eval": use_cot_prompt_eval,
        "prepare_caches": prepare_caches,
        "cache_logits": cache_logits,
        "cache_cot": cache_cot,
        "use_logits_kd": use_logits_kd,
        "allow_insufficient_runs": allow_insufficient_runs,
        "hypothesis_metric": hypothesis_metric,
        "student_key": student_key,
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
            )
            state["teacher_cot_file"] = str(cot_file)

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
    teacher_cot_file = state.get("teacher_cot_file")

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

            # Load models
            teacher_model = None
            teacher_tok = None
            if cond_cfg["mode"] == "traditional" and not teacher_logits_traditional_dir and use_logits_kd:
                teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)
            if cond_cfg["mode"] == "reasoning" and not teacher_cot_file:
                teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)
            if cond_cfg["mode"] == "reasoning" and use_logits_kd and not teacher_logits_reasoning_dir:
                # Only needed if logits-KD is requested for reasoning mode.
                teacher_model, teacher_tok = setup_model_and_tokenizer(cfg.model_hierarchy["teacher_medium"], cfg.device, cfg.quantization)

            student_name = cfg.model_hierarchy[student_key]
            student_model, student_tok = setup_model_and_tokenizer(student_name, cfg.device, cfg.quantization)

            # Scientific validity fix: logits-KD requires tokenizer compatibility.
            if use_logits_kd:
                cache_dir = teacher_logits_traditional_dir if cond_cfg["mode"] == "traditional" else teacher_logits_reasoning_dir
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
                    if teacher_tok is None:
                        # Load teacher tokenizer just for compatibility check.
                        teacher_tok = AutoTokenizer.from_pretrained(cfg.model_hierarchy["teacher_medium"])
                    assert_tokenizer_compatible_for_logits_kd(
                        teacher_tok,
                        student_tok,
                        context=f"mode={cond_cfg['mode']}, student={student_key}",
                    )

            # Data
            train_ds = load_training_dataset(enable_gsm8k_train, enable_bbh_train, cfg.train_limit, seed=seed)

            # Scientific validity fix: reasoning condition requires teacher-generated CoT
            # (cached or generated on demand). No silent fallback to gold answers.
            if cond_cfg["mode"] == "reasoning" and not teacher_cot_file:
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
                    )
                )
                state["teacher_cot_file"] = str(teacher_cot_file)
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
            else:
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
                )

            # Eval
            eval_results = evaluator.evaluate(
                trained_model,
                student_tok,
                seed=seed,
                eval_gsm8k=eval_gsm8k,
                eval_bbh=eval_bbh,
                eval_efficiency=eval_efficiency,
                use_cot_prompt=use_cot_prompt_eval,
                generation_cfg=cfg.eval_generation,
            )

            # Save model (single save)
            save_dir = cfg.models_dir / f"{cond_name}_seed{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)
            trained_model.save_pretrained(save_dir)
            student_tok.save_pretrained(save_dir)
            print(f" Modelo salvo em: {save_dir}")

            run_payload = {
                "seed": seed,
                "condition": cond_name,
                "description": cond_cfg["description"],
                "training": train_metrics,
                "evaluation": eval_results,
            }

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = cfg.reports_dir / f"comprehensive_report_{exp_id}_{timestamp}.json"
    summary_txt = cfg.reports_dir / f"results_summary_{exp_id}_{timestamp}.txt"
    write_report_json(report_json, results)
    write_summary_txt(summary_txt, results)

    print(" Relatórios salvos:")
    print(f"   - JSON: {report_json}")
    print(f"   - TXT:  {summary_txt}")

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="H1 experiment: KD traditional vs CoT-aware KD")
    p.add_argument("--kd_modes", nargs="+", default=["traditional", "reasoning"], choices=["traditional", "reasoning"])
    p.add_argument("--student", default="student_primary", choices=["student_primary", "student_small"])

    p.add_argument("--seed", type=int, action="append", dest="seeds", help="Repeatable. Example: --seed 42 --seed 43")
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--max_length", type=int, default=512)

    # Scientific control fix: explicit enable/disable flags; no default=True store_true.
    p.add_argument("--enable_gsm8k_train", action="store_true", help="Use GSM8K train split for training")
    p.add_argument("--disable_gsm8k_train", action="store_true", help="Disable GSM8K training")
    p.add_argument("--enable_bbh_train", action="store_true", help="Enable BBH/BBEH-derived training split (non-overlapping with eval)")
    p.add_argument("--disable_bbh_train", action="store_true", help="Disable BBH training")

    p.add_argument("--eval_gsm8k", action="store_true", help="Evaluate on GSM8K test")
    p.add_argument("--no_eval_gsm8k", action="store_true", help="Disable GSM8K evaluation")
    p.add_argument("--eval_bbh", action="store_true", help="Evaluate on BBEH/BBH held-out split")
    p.add_argument("--no_eval_bbh", action="store_true", help="Disable BBH evaluation")
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

    p.add_argument("--eval_max_new_tokens", type=int, default=256)
    p.add_argument("--eval_temperature", type=float, default=0.0)

    p.add_argument("--teacher_cot_max_new_tokens", type=int, default=150)
    p.add_argument("--teacher_cot_temperature", type=float, default=0.0)

    return p


def main(argv: Optional[Sequence[str]] = None):
    args = build_arg_parser().parse_args(argv)
    cfg = EvidenceBasedConfig()

    if args.seeds:
        cfg.seeds = list(args.seeds)
    if args.train_limit is not None:
        cfg.train_limit = args.train_limit
    cfg.max_length = args.max_length

    cfg.eval_generation = GenerationConfig(max_new_tokens=args.eval_max_new_tokens, temperature=args.eval_temperature, do_sample=False)
    cfg.teacher_cot_generation = GenerationConfig(
        max_new_tokens=args.teacher_cot_max_new_tokens, temperature=args.teacher_cot_temperature, do_sample=False
    )

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

    return run_experiment(
        cfg,
        kd_modes=args.kd_modes,
        enable_gsm8k_train=enable_gsm8k_train,
        enable_bbh_train=enable_bbh_train,
        eval_gsm8k=eval_gsm8k,
        eval_bbh=eval_bbh,
        eval_efficiency=eval_eff,
        use_cot_prompt_eval=use_cot_eval,
        prepare_caches=prepare_caches,
        cache_logits=cache_logits,
        cache_cot=cache_cot,
        use_logits_kd=use_logits_kd,
        allow_insufficient_runs=bool(args.allow_insufficient_runs),
        hypothesis_metric=str(args.hypothesis_metric),
        student_key=str(args.student),
    )


if __name__ == "__main__":
    main()
