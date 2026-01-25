from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


def _default_drive_root() -> Path:
    env = os.environ.get("SLM_DRIVE_ROOT")
    if env:
        return Path(env)

    # Colab convention.
    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.exists():
        return colab_drive / "SLM_results"

    # Fallbacks for non-Colab runs (e.g., local dev/Windows).
    if os.name == "nt":
        return Path.cwd() / "SLM_results"
    if Path("/content").exists():
        return Path("/content/SLM_results")
    return Path.cwd() / "SLM_results"


def _is_colab_drive_mounted() -> bool:
    # In Colab, /content/drive is a mount point when Drive is mounted.
    try:
        return os.path.ismount("/content/drive") and Path("/content/drive/MyDrive").is_dir()
    except Exception:
        return False


@dataclass(frozen=True)
class GenerationConfig:
    """Explicit generation parameters to make runs & caches traceable."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = 1.1

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBasedConfig:
    """Centralized configuration for the experimental core."""

    model_hierarchy: Dict[str, str] = field(
        default_factory=lambda: {
            # Default to a single family to keep logits-KD scientifically valid
            # (token IDs must match across teacher/student).
            "teacher_medium": "Qwen/Qwen2.5-7B-Instruct",
            "student_primary": "Qwen/Qwen2.5-3B-Instruct",
            "student_small": "Qwen/Qwen2.5-1.5B-Instruct",
        }
    )

    # Core experiment knobs
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    max_length: int = 512

    # Limits (ajustados para resultados científicos mais robustos)
    # GSM8K train tem ~7.5k exemplos; usar 5000 cobre ~67% do dataset
    train_limit: Optional[int] = 5000  # Antes: 3000 - aumentado para melhor generalização
    # Avaliação com mais exemplos reduz variância e aumenta confiabilidade estatística
    eval_limit_gsm8k: int = 300  # Antes: 100 - GSM8K test tem 1319 exemplos
    eval_limit_bbh: int = 100   # Antes: 30 - mais exemplos por task
    # OOD commonsense (eval-only; not part of primary hypothesis by default)
    eval_limit_obqa: int = 200

    # Distillation hyperparams (otimizado para Google Colab A100 40/80GB)
    # Ajustes de estabilidade:
    #   - temperature_schedule reduzido: evita amplificação excessiva da KL divergence
    #   - alpha_schedule reduzido: mais peso na CE (ground-truth) para estabilidade
    #   - clip_grad_norm mais agressivo: previne explosão de gradientes
    #   - batch_size aumentado para A100 (80GB suporta 8, 40GB suporta 4-6)
    kd_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "temperature_schedule": [2.0, 1.5, 1.0],  # Suavizado para estabilidade
            "alpha_schedule": [0.3, 0.2, 0.1],  # REDUZIDO: mais peso na CE (student aprende do texto, não só logits)
            "learning_rates": {"kd": 5e-5},  # AUMENTADO: 3e-5 era muito conservador
            "lora_rank": 32,  # AUMENTADO: mais capacidade de aprendizado (antes: 16)
            "epochs": 4,  # AUMENTADO: mais tempo para convergir (antes: 3)
            # A100: 80GB suporta batch=8, 40GB suporta batch=4-6
            # Qwen vocab (~150k) continua exigindo cuidado com (B,T,V) tensors
            "batch_size": 4,  # A100 permite mais
            "grad_accum_steps": 4,  # Effective batch = 16
            "clip_grad_norm": 1.0,  # Relaxado um pouco (antes: 0.5 era muito agressivo)
            "dataloader_num_workers": 2,  # Antes: 0 - A100 tem CPU suficiente
        }
    )

    # Experimental design
    seeds: List[int] = field(default_factory=lambda: [42])
    bootstrap_samples: int = 5000
    alpha_level: float = 0.05

    # Quantization (kept compatible with Colab)
    quantization: Dict[str, Any] = field(
        default_factory=lambda: {
            "load_in_4bit": False,
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "device_map": "auto",
        }
    )

    # Output
    drive_root: Path = field(default_factory=_default_drive_root)
    output_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    experiments_dir: Path = field(init=False)

    # Defaults for the hypothesis H1 experiment
    # NOTA: max_new_tokens do teacher aumentado para garantir que o modelo
    # consiga gerar raciocínio completo + marcador ### FINAL_ANSWER: + resposta
    teacher_cot_generation: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=256, temperature=0.0, do_sample=False)
    )
    eval_generation: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=192, temperature=0.0, do_sample=False)
    )

    # Hypothesis (kept as documentation metadata)
    scientific_hypotheses: Dict[str, str] = field(
        default_factory=lambda: {
            "main": (
                "H1: Knowledge Distillation com superviso de raciocnio (CoT) "
                "melhora o desempenho do student em tarefas de raciocnio "
                "em relao  distilao tradicional (answer-only)."
            )
        }
    )

    def __post_init__(self) -> None:
        self.drive_root = Path(self.drive_root)

        # If user points to Colab Drive but it isn't mounted, don't silently
        # pretend it is Drive (it becomes ephemeral storage under /content).
        drive_str = str(self.drive_root)
        if drive_str.startswith("/content/drive") and not _is_colab_drive_mounted():
            fallback = Path("/content/SLM_results")
            print(
                "[WARN] Google Drive não parece estar montado em /content/drive. "
                f"Resultados NÃO serão sincronizados com o Drive. Salvando em: {fallback}. "
                "Para salvar no Drive: from google.colab import drive; drive.mount('/content/drive')"
            )
            self.drive_root = fallback

        self.output_dir = self.drive_root / "scientific_results"
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.experiments_dir = self.output_dir / "experiments"
        for path in (
            self.output_dir,
            self.models_dir,
            self.reports_dir,
            self.experiments_dir,
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Falha ao criar diretório de saída: {path}. Erro: {e}")

    def to_metadata(self) -> Dict[str, Any]:
        """Metadata snapshot for reports and run traceability."""

        return {
            "timestamp": datetime.now().isoformat(),
            "output_paths": {
                "drive_root": str(self.drive_root),
                "output_dir": str(self.output_dir),
                "models_dir": str(self.models_dir),
                "reports_dir": str(self.reports_dir),
                "experiments_dir": str(self.experiments_dir),
            },
            "model_hierarchy": dict(self.model_hierarchy),
            "max_length": self.max_length,
            "train_limit": self.train_limit,
            "eval_limits": {
                "gsm8k": self.eval_limit_gsm8k,
                "bbh": self.eval_limit_bbh,
                "obqa": self.eval_limit_obqa,
            },
            "kd_params": json.loads(json.dumps(self.kd_params, default=str)),
            "seeds": list(self.seeds),
            "bootstrap_samples": self.bootstrap_samples,
            "alpha_level": self.alpha_level,
            "quantization": json.loads(json.dumps(self.quantization, default=str)),
            "teacher_cot_generation": self.teacher_cot_generation.to_jsonable(),
            "eval_generation": self.eval_generation.to_jsonable(),
            "hypotheses": dict(self.scientific_hypotheses),
        }


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[torch.device | str]) -> torch.device:
    dev_str = str(device or "cpu")
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        print(" CUDA indisponvel  usando CPU.")
        dev_str = "cpu"
    return torch.device(dev_str)


def safe_model_to(model: Optional[torch.nn.Module], device: Optional[torch.device | str]):
    if model is None or device is None:
        return model

    target = resolve_device(device)
    if getattr(model, "hf_device_map", None):
        return model
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        return model

    first_param = next(model.parameters(), None)
    if first_param is None:
        return model
    if getattr(first_param, "is_meta", False):
        print(" Modelo em meta device; pule .to() e configure device_map manualmente.")
        return model
    if first_param.device == target:
        return model
    return model.to(target)


def get_safe_tokenizer_length(tokenizer, fallback: int = 2048, upper_bound: int = 4096) -> int:
    raw_max = getattr(tokenizer, "model_max_length", None) or fallback
    try:
        raw_int = int(raw_max)
    except (TypeError, ValueError, OverflowError):
        raw_int = fallback
    if raw_int <= 0:
        raw_int = fallback
    return min(raw_int, upper_bound)


def ensure_tokenizer_has_pad(tokenizer, model=None, pad_token: str = "<|pad|>") -> bool:
    if tokenizer.pad_token_id is not None:
        return False
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        return False
    tokenizer.add_special_tokens({"pad_token": pad_token})
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
    return True


def get_schedule_value(schedule: Optional[Sequence[float]], epoch_idx: int, default: float) -> float:
    if schedule is None:
        return default
    if not schedule:
        return default
    index = min(max(epoch_idx, 0), len(schedule) - 1)
    try:
        return float(schedule[index])
    except Exception:
        return default
