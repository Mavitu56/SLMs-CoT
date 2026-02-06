"""CLI argument parsing para run_experiment.py."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos para experimentos."""
    p = argparse.ArgumentParser(description="H1 experiment: KD traditional vs CoT-aware KD")
    
    # Diretórios e configuração básica
    p.add_argument("--drive_root", type=str, default=None,
        help="Output root directory. In Colab, use /content/drive/MyDrive/SLM_results")
    p.add_argument("--kd_modes", nargs="+", default=["traditional", "reasoning"],
        choices=["traditional", "reasoning", "cascod", "combo_d"])
    p.add_argument("--student", default="student_primary", 
        choices=["student_primary", "student_small"])

    # Seeds e limites
    p.add_argument("--seed", type=int, action="append", dest="seeds",
        help="Repeatable. Example: --seed 42 --seed 43")
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--max_length", type=int, default=512)

    # Hiperparâmetros de treinamento
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)

    # Quantização
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_load_in_4bit", action="store_true")

    # Controle de datasets de treino
    p.add_argument("--enable_gsm8k_train", action="store_true")
    p.add_argument("--disable_gsm8k_train", action="store_true")
    p.add_argument("--enable_bbh_train", action="store_true")
    p.add_argument("--disable_bbh_train", action="store_true")

    # Avaliação
    p.add_argument("--eval_gsm8k", action="store_true")
    p.add_argument("--no_eval_gsm8k", action="store_true")
    p.add_argument("--eval_limit_gsm8k", type=int, default=None)
    p.add_argument("--eval_bbh", action="store_true")
    p.add_argument("--no_eval_bbh", action="store_true")
    p.add_argument("--eval_limit_bbh", type=int, default=None)
    p.add_argument("--eval_obqa", action="store_true")
    p.add_argument("--no_eval_obqa", action="store_true")
    p.add_argument("--eval_limit_obqa", type=int, default=200)
    p.add_argument("--eval_efficiency", action="store_true")
    p.add_argument("--no_eval_efficiency", action="store_true")

    # Prompt de avaliação
    p.add_argument("--use_cot_prompt_eval", action="store_true")
    p.add_argument("--no_cot_prompt_eval", action="store_true")

    # KD por logits
    p.add_argument("--use_logits_kd", action="store_true")
    p.add_argument("--no_logits_kd", action="store_true")
    p.add_argument("--allow_insufficient_runs", action="store_true")

    # Controles de reprodutibilidade
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no_deterministic", action="store_true")
    p.add_argument("--eval_protocol_strict", action="store_true")
    p.add_argument("--no_eval_protocol_strict", action="store_true")

    # Reasoning mask
    p.add_argument("--reasoning_mask_strict", action="store_true")
    p.add_argument("--no_reasoning_mask_strict", action="store_true")
    p.add_argument("--reasoning_mask_fallback_to_completion", action="store_true")
    p.add_argument("--no_reasoning_mask_fallback_to_completion", action="store_true")
    p.add_argument("--reasoning_mask_max_fallback", type=float, default=None)
    p.add_argument("--reasoning_mask_min_reasoning_frac", type=float, default=None)

    # Sanitização de logits
    p.add_argument("--train_sanitize_logits", action="store_true")
    p.add_argument("--no_train_sanitize_logits", action="store_true")
    p.add_argument("--train_max_logit_abs", type=float, default=None)

    p.add_argument("--hypothesis_metric", default="primary_score",
        choices=["primary_score", "overall_score"])

    # CasCoD
    p.add_argument("--cascod_stage1_epochs", type=int, default=1)
    p.add_argument("--cascod_stage2_epochs", type=int, default=-1)
    p.add_argument("--cascod_alpha", type=float, default=0.3)
    p.add_argument("--cascod_filter_by_gold", dest="cascod_filter_by_gold", action="store_true")
    p.add_argument("--no_cascod_filter_by_gold", dest="cascod_filter_by_gold", action="store_false")
    p.set_defaults(cascod_filter_by_gold=True)

    # Geração de tokens
    p.add_argument("--eval_max_new_tokens", type=int, default=192)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--teacher_cot_max_new_tokens", type=int, default=128)
    p.add_argument("--teacher_cot_temperature", type=float, default=0.0)

    # Granularidade
    p.add_argument("--granularity_level", type=int, default=0)
    p.add_argument("--granularity_one_shot", dest="granularity_one_shot", action="store_true")
    p.add_argument("--no_granularity_one_shot", dest="granularity_one_shot", action="store_false")
    p.set_defaults(granularity_one_shot=None)
    p.add_argument("--granularity_multi_level", dest="granularity_multi_level", action="store_true")
    p.add_argument("--no_granularity_multi_level", dest="granularity_multi_level", action="store_false")
    p.set_defaults(granularity_multi_level=None)

    # Post-CoT
    p.add_argument("--post_cot", dest="post_cot", action="store_true")
    p.add_argument("--no_post_cot", dest="post_cot", action="store_false")
    p.set_defaults(post_cot=False)
    p.add_argument("--post_cot_gold_rationale", dest="post_cot_gold_rationale", action="store_true")
    p.add_argument("--no_post_cot_gold_rationale", dest="post_cot_gold_rationale", action="store_false")
    p.set_defaults(post_cot_gold_rationale=None)
    p.add_argument("--post_cot_use_ig", dest="post_cot_use_ig", action="store_true")
    p.add_argument("--no_post_cot_use_ig", dest="post_cot_use_ig", action="store_false")
    p.set_defaults(post_cot_use_ig=None)
    p.add_argument("--post_cot_ig_steps", type=int, default=8)
    p.add_argument("--post_cot_ig_top_frac", type=float, default=0.3)

    # Baselines
    p.add_argument("--ft_teacher", action="store_true",
        help="Baseline 0.1: SFT teacher")
    p.add_argument("--ft_student", action="store_true",
        help="Baseline 0.2: SFT student (no KD)")
    p.add_argument("--kd_logits_baseline", action="store_true",
        help="Baseline 0.3: KD por logits")
    p.add_argument("--teacher_ckpt_dir", type=str, default=None,
        help="Diretório do checkpoint do teacher")
    p.add_argument("--kd_cot_baseline", action="store_true",
        help="Baseline 0.4: KD CoT padrão")

    return p


def resolve_bool_flags(args, enable_attr: str, disable_attr: str, default: bool) -> bool:
    """Resolve par de flags booleanas enable/disable."""
    if getattr(args, enable_attr, False):
        return True
    if getattr(args, disable_attr, False):
        return False
    return default
