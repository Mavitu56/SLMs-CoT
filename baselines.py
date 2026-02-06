"""Funcionalidades consolidadas de baseline para run_experiment.py."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from config import EvidenceBasedConfig, set_seed
from data import load_training_dataset
from report import write_plots, write_report_json, write_summary_txt
from utils import env_flag


def generate_experiment_reports(
    cfg: EvidenceBasedConfig,
    exp_id: str,
    exp_dir: Path,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """Gera relatórios JSON, TXT e gráficos para um experimento.
    
    Consolida o padrão repetido em todas as funções baseline.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_json = cfg.reports_dir / f"comprehensive_report_{exp_id}_{timestamp}.json"
    summary_txt = cfg.reports_dir / f"results_summary_{exp_id}_{timestamp}.txt"

    results.setdefault("artifacts", {})
    results["artifacts"].update({
        "experiment_id": exp_id,
        "experiment_dir": str(exp_dir),
        "report_json": str(report_json),
        "summary_txt": str(summary_txt),
    })

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


def cleanup_model(model) -> None:
    """Libera memória de GPU após uso do modelo."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def should_skip_run(state: Dict[str, Any], run_key: str, cond_runs: List[Dict[str, Any]]) -> bool:
    """Verifica se um run já foi completado e deve ser pulado."""
    if state.get("completed", {}).get(run_key):
        print(f" Pulando run já completado: {run_key}")
        cond_runs.append(state["completed"][run_key])
        return True
    return False


def save_run_state(
    state: Dict[str, Any],
    state_path: Path,
    run_key: str,
    run_payload: Dict[str, Any],
    artifacts: Optional[Dict[str, str]] = None,
) -> None:
    """Salva estado de um run no state.json."""
    import json
    
    state.setdefault("completed", {})[run_key] = run_payload
    if artifacts:
        state.setdefault("artifacts", {}).update(artifacts)
    
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def create_run_payload(
    seed: int,
    cond_name: str,
    description: str,
    train_metrics: Dict[str, Any],
    eval_results: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    """Cria payload padronizado de um run."""
    return {
        "seed": seed,
        "condition": cond_name,
        "description": description,
        "training": train_metrics,
        "evaluation": eval_results,
        "artifacts": artifacts,
    }


def finalize_condition_results(
    results: Dict[str, Any],
    cond_name: str,
    description: str,
    cond_runs: List[Dict[str, Any]],
) -> None:
    """Finaliza results com os runs de uma condição."""
    results["conditions"][cond_name] = {
        "description": description,
        "runs": cond_runs,
    }
