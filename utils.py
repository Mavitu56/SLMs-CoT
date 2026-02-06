"""Utilitários compartilhados para evitar duplicação de código."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================================
# Environment helpers (consolidados de distill.py, run_experiment.py)
# ============================================================================

def env_flag(name: str, default: str = "0") -> bool:
    """Lê flag booleana de variável de ambiente."""
    v = os.environ.get(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    """Lê float de variável de ambiente."""
    v = os.environ.get(name)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    """Lê int de variável de ambiente."""
    v = os.environ.get(name)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


# ============================================================================
# Logit sanitization (consolidado de distill.py)
# ============================================================================

def sanitize_logits(x: torch.Tensor, max_abs: float = 50.0, enabled: bool = True) -> torch.Tensor:
    """Sanitiza logits removendo NaN/inf para estabilidade numérica."""
    if not enabled:
        return x
    x = torch.nan_to_num(x, nan=0.0, posinf=max_abs, neginf=-max_abs)
    if max_abs > 0:
        x = x.clamp(min=-max_abs, max=max_abs)
    return x


def has_nonfinite(x: torch.Tensor) -> bool:
    """Verifica se tensor contém valores não-finitos."""
    try:
        return bool((~torch.isfinite(x)).any().item())
    except Exception:
        return True


# ============================================================================
# Environment metadata (consolidado de run_experiment.py, evaluate_saved_models.py)
# ============================================================================

def collect_environment_metadata() -> Dict[str, Any]:
    """Coleta metadados do ambiente de execução."""
    meta: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    try:
        meta["torch_version"] = getattr(torch, "__version__", None)
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["cuda_version"] = getattr(torch.version, "cuda", None)
        try:
            meta["cudnn_version"] = int(torch.backends.cudnn.version() or 0)
        except Exception:
            meta["cudnn_version"] = None
        if bool(torch.cuda.is_available()):
            try:
                meta["gpu_name"] = str(torch.cuda.get_device_name(0))
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


# ============================================================================
# Tokenizer fingerprint (de run_experiment.py)
# ============================================================================

def tokenizer_fingerprint(tokenizer) -> str:
    """Computa fingerprint estável para verificar compatibilidade de tokenizers."""
    try:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    except Exception:
        vocab = None

    h = hashlib.sha256()
    h.update(str(getattr(tokenizer, "__class__", type(tokenizer)).__name__).encode("utf-8"))
    h.update(b"\n")
    
    for attr in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        h.update(f"{attr}={getattr(tokenizer, attr, None)}\n".encode("utf-8"))

    if isinstance(vocab, dict) and vocab:
        try:
            for token, idx in sorted(vocab.items(), key=lambda kv: (kv[0], kv[1])):
                h.update(token.encode("utf-8", errors="ignore"))
                h.update(b"=")
                h.update(str(int(idx)).encode("utf-8"))
                h.update(b"\n")
        except Exception:
            h.update(f"vocab_len={len(vocab)}\n".encode("utf-8"))

    return h.hexdigest()[:16]


# ============================================================================
# Model loading (consolidado de run_experiment.py, evaluate_saved_models.py)
# ============================================================================

def load_model_and_tokenizer(
    model_path: Path | str,
    device: torch.device,
    quant_cfg: Optional[Dict[str, Any]] = None,
    load_dtype: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Carrega modelo HF completo ou adapter PEFT de um diretório.
    
    Args:
        model_path: Caminho para diretório do modelo ou adapter
        device: Device para carregar o modelo
        quant_cfg: Configuração de quantização (opcional)
        load_dtype: Tipo de dado ('bf16', 'fp16', 'fp32', 'auto')
    
    Returns:
        Tuple (model, tokenizer)
    """
    from config import ensure_tokenizer_has_pad, get_safe_tokenizer_length, safe_model_to
    
    model_dir = Path(model_path)
    quant_cfg = quant_cfg or {}
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.model_max_length = get_safe_tokenizer_length(tokenizer, fallback=2048, upper_bound=4096)
    tokenizer.padding_side = "left"
    tokenizer_len = len(tokenizer)

    model_kwargs: Dict[str, Any] = {}
    
    # Configuração de dtype
    if load_dtype and load_dtype != "auto":
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if load_dtype in dtype_map:
            model_kwargs["torch_dtype"] = dtype_map[load_dtype]
    
    # Configuração de quantização 4-bit
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

    # Modelo completo (config.json)
    if (model_dir / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        if not bool(quant_cfg.get("load_in_4bit")):
            model = safe_model_to(model, device)
        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()
        try:
            setattr(model, "name_or_path", str(model_dir))
        except Exception:
            pass
        return model, tokenizer

    # Adapter PEFT (adapter_config.json)
    if (model_dir / "adapter_config.json").exists():
        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
        if not base_name:
            raise ValueError(
                f"adapter_config.json em {model_dir} não contém 'base_model_name_or_path'."
            )

        base_model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)

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
            raise RuntimeError(f"Falha ao carregar adapter PEFT: {exc}")

        if not bool(quant_cfg.get("load_in_4bit")):
            model = safe_model_to(model, device)
        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()
        return model, tokenizer

    raise ValueError(f"Diretório não é modelo HF nem adapter PEFT: {model_dir}")


def iter_model_dirs(paths: Sequence[str], recursive: bool = False):
    """Itera diretórios de modelos a partir de caminhos."""
    out = []
    for p in paths:
        root = Path(p)
        if not root.exists():
            continue
        if root.is_dir() and ((root / "config.json").exists() or (root / "adapter_config.json").exists()):
            out.append(root)
            continue
        if root.is_dir():
            for cand in root.glob("*/config.json"):
                out.append(cand.parent)
            for cand in root.glob("*/adapter_config.json"):
                out.append(cand.parent)
        if root.is_dir() and recursive:
            for cand in root.rglob("config.json"):
                out.append(cand.parent)
            for cand in root.rglob("adapter_config.json"):
                out.append(cand.parent)

    seen = set()
    uniq = []
    for d in out:
        k = str(d.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    return uniq


# ============================================================================
# JSON helpers
# ============================================================================

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Escreve dicionário como JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def now_stamp() -> str:
    """Retorna timestamp formatado para nomes de arquivo."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
