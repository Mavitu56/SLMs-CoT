from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import EvidenceBasedConfig, GenerationConfig, ensure_tokenizer_has_pad, resolve_device, set_seed
from eval import StandardizedEvaluator


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _iter_model_dirs(paths: Sequence[str], recursive: bool) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        root = Path(p)
        if not root.exists():
            continue
        if root.is_dir() and ((root / "config.json").exists() or (root / "adapter_config.json").exists()):
            out.append(root)
            continue

        if root.is_dir():
            # Common case: a root directory containing many model subfolders.
            for cand in root.glob("*/config.json"):
                out.append(cand.parent)
            for cand in root.glob("*/adapter_config.json"):
                out.append(cand.parent)

        if root.is_dir() and recursive:
            for cand in root.rglob("config.json"):
                out.append(cand.parent)
            for cand in root.rglob("adapter_config.json"):
                out.append(cand.parent)

    # De-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for d in out:
        k = str(d.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(d)
    return uniq


def _load_model_and_tokenizer(model_dir: Path, device: torch.device, load_dtype: str) -> Tuple[Any, Any]:
    """Load either a full HF model dir (config.json) or a PEFT adapter dir (adapter_config.json)."""

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer_len = len(tokenizer)

    model_kwargs: Dict[str, Any] = {}
    if load_dtype and load_dtype != "auto":
        if load_dtype == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif load_dtype == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif load_dtype == "fp32":
            model_kwargs["torch_dtype"] = torch.float32

    if (model_dir / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        model = model.to(device)
        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()
        return model, tokenizer

    if (model_dir / "adapter_config.json").exists():
        # Adapter-only save (PEFT/LoRA). Load base model and attach adapter.
        import json

        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
        if not base_name:
            raise ValueError(
                f"adapter_config.json em {model_dir} não contém 'base_model_name_or_path'. "
                "Não é possível carregar o modelo base para avaliação."
            )

        base_model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)

        # If the adapter was saved with resized embeddings (PEFT may do this when
        # tokenizer/model vocab sizes diverge), ensure the base model matches the
        # tokenizer *before* loading adapter weights.
        try:
            emb = base_model.get_input_embeddings()
            if emb is not None and tokenizer_len > int(emb.weight.shape[0]):
                base_model.resize_token_embeddings(tokenizer_len)
        except Exception:
            # Best-effort; if resize isn't supported, loading may still fail.
            pass
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, model_dir)
        except Exception as exc:
            raise RuntimeError(
                "Falha ao carregar adapter PEFT. Instale 'peft' e verifique compatibilidade do adapter com o modelo base. "
                f"Erro: {exc}"
            )

        model = model.to(device)
        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()
        return model, tokenizer

    raise ValueError(f"Diretório não parece ser um modelo HF nem um adapter PEFT: {model_dir}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate models saved by this repo (runs StandardizedEvaluator with consistent inference settings)."
    )

    p.add_argument(
        "--model_dir",
        action="append",
        dest="model_dirs",
        default=[],
        help="Path to a saved model directory (contains config.json). Repeatable.",
    )
    p.add_argument(
        "--models_root",
        type=str,
        default=None,
        help="Root folder containing many saved models; use with --recursive.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="When --models_root is provided, find all nested model dirs (config.json).",
    )

    p.add_argument("--seed", type=int, default=42, help="Seed for eval sampling and generation reproducibility.")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--eval_limit_gsm8k", type=int, default=200)
    p.add_argument("--eval_limit_bbh", type=int, default=50)

    p.add_argument("--eval_gsm8k", action="store_true")
    p.add_argument("--no_eval_gsm8k", action="store_true")
    p.add_argument("--eval_bbh", action="store_true")
    p.add_argument("--no_eval_bbh", action="store_true")
    p.add_argument("--eval_efficiency", action="store_true")
    p.add_argument("--no_eval_efficiency", action="store_true")

    p.add_argument("--use_cot_prompt_eval", action="store_true")
    p.add_argument("--no_cot_prompt_eval", action="store_true")

    p.add_argument(
        "--cascod_two_stage_eval",
        action="store_true",
        help="If set, run CasCoD-style two-stage inference (q→r then q,r→a) during eval.",
    )
    p.add_argument(
        "--no_cascod_two_stage_eval",
        action="store_true",
        help="Disable CasCoD two-stage inference during eval.",
    )

    p.add_argument("--eval_max_new_tokens", type=int, default=256)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument(
        "--eval_do_sample",
        action="store_true",
        help="Use sampling during eval (default: False/greedy).",
    )

    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., cuda, cuda:0, cpu). Default: auto.",
    )
    p.add_argument(
        "--load_dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Optional dtype override when loading model.",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If set, write eval JSONs here instead of next to each model.",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Resolve model dirs
    model_paths: List[str] = list(args.model_dirs or [])
    if args.models_root:
        model_paths.append(args.models_root)

    model_dirs = _iter_model_dirs(model_paths, recursive=bool(args.recursive))

    # Friendly fallback: if user passed a models_root but forgot --recursive, try it.
    if not model_dirs and args.models_root and not args.recursive:
        print("[eval] Nenhum model_dir encontrado sem --recursive; tentando busca recursiva...")
        model_dirs = _iter_model_dirs(model_paths, recursive=True)

    if not model_dirs:
        debug_lines = ["Nenhum model_dir encontrado.", "Paths inspecionados:"]
        for p in model_paths:
            root = Path(p)
            debug_lines.append(f"  - {root} (exists={root.exists()}, is_dir={root.is_dir()})")
        debug_lines.append("")
        debug_lines.append("Dicas:")
        debug_lines.append("  - Se você passou uma pasta raiz com vários modelos, use --models_root <pasta> --recursive")
        debug_lines.append("  - Um model_dir válido é uma pasta que contém config.json (save_pretrained do HF)")
        raise SystemExit("\n".join(debug_lines))

    # Flags (match run_experiment defaults unless overridden)
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

    use_cot = True
    if args.use_cot_prompt_eval:
        use_cot = True
    if args.no_cot_prompt_eval:
        use_cot = False

    cascod_two_stage = False
    if args.cascod_two_stage_eval:
        cascod_two_stage = True
    if args.no_cascod_two_stage_eval:
        cascod_two_stage = False

    device = resolve_device(args.device)

    # Build evaluator config consistent with the repo.
    cfg = EvidenceBasedConfig()
    cfg.device = device
    cfg.max_length = int(args.max_length)
    cfg.eval_limit_gsm8k = int(args.eval_limit_gsm8k)
    cfg.eval_limit_bbh = int(args.eval_limit_bbh)
    cfg.eval_generation = GenerationConfig(
        max_new_tokens=int(args.eval_max_new_tokens),
        temperature=float(args.eval_temperature),
        do_sample=bool(args.eval_do_sample),
    )

    evaluator = StandardizedEvaluator(cfg)

    out_root = Path(args.out_dir) if args.out_dir else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))

    summary_rows: List[Dict[str, Any]] = []

    for model_dir in model_dirs:
        print(f"\n[eval] Loading model: {model_dir}")
        model, tokenizer = _load_model_and_tokenizer(model_dir, device=device, load_dtype=str(args.load_dtype))

        print(
            f"[eval] Running evaluator flags: gsm8k={eval_gsm8k}, bbh={eval_bbh}, efficiency={eval_eff}, cot_prompt={use_cot}, cascod_two_stage={cascod_two_stage}"
        )
        results = evaluator.evaluate(
            model,
            tokenizer,
            seed=int(args.seed),
            eval_gsm8k=eval_gsm8k,
            eval_bbh=eval_bbh,
            eval_efficiency=eval_eff,
            use_cot_prompt=use_cot,
            cascod_two_stage=bool(cascod_two_stage),
            generation_cfg=cfg.eval_generation,
        )

        artifact = {
            "timestamp": datetime.now().isoformat(),
            "model_dir": str(model_dir),
            "device": str(device),
            "seed": int(args.seed),
            "eval_flags": {
                "eval_gsm8k": eval_gsm8k,
                "eval_bbh": eval_bbh,
                "eval_efficiency": eval_eff,
                "use_cot_prompt_eval": use_cot,
                "cascod_two_stage_eval": bool(cascod_two_stage),
            },
            "generation": cfg.eval_generation.to_jsonable(),
            "limits": {
                "max_length": int(cfg.max_length),
                "eval_limit_gsm8k": int(cfg.eval_limit_gsm8k),
                "eval_limit_bbh": int(cfg.eval_limit_bbh),
            },
            "results": results,
        }

        stamp = _now_stamp()
        out_dir = out_root if out_root else model_dir
        out_path = out_dir / f"eval_{stamp}.json"
        _write_json(out_path, artifact)
        print(f"[eval] Saved: {out_path}")

        # Small one-line summary
        gsm = float(results.get("gsm8k", {}).get("accuracy", 0.0) or 0.0)
        bbh = float(results.get("bbh", {}).get("average_accuracy", 0.0) or 0.0)
        primary = float(results.get("primary_score", 0.0) or 0.0)
        overall = float(results.get("overall_score", 0.0) or 0.0)
        print(f"[eval] Summary: primary={primary:.4f} overall={overall:.4f} gsm8k={gsm:.4f} bbh={bbh:.4f}")

        summary_rows.append(
            {
                "model_dir": str(model_dir),
                "eval_json": str(out_path),
                "primary_score": primary,
                "overall_score": overall,
                "gsm8k_accuracy": gsm,
                "bbh_average_accuracy": bbh,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Global summary
    if out_root:
        summary_path = out_root / f"eval_summary_{_now_stamp()}.json"
        _write_json(summary_path, {"rows": summary_rows})
        print(f"\n[eval] Wrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
