from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ScientificLogger:
    logs: List[Dict[str, Any]]
    hyperparameters: Dict[str, Any]

    def __init__(self):
        self.logs = []
        self.hyperparameters = {}

    def log_phase(self, phase: str, payload: Dict[str, Any]) -> None:
        entry = {"timestamp": datetime.now().isoformat(), "phase": phase, "payload": payload}
        self.logs.append(entry)
        print(f"[SCI-LOG - {phase}] -> {json.dumps(payload, default=str)[:200]}...")

    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        self.hyperparameters.update(hyperparams)
        self.log_phase("HYPERPARAMETERS", hyperparams)


def write_report_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_summary_txt(path: Path, results: Dict[str, Any]) -> None:
    lines = ["=== RESUMO DE RESULTADOS ===", f"Data: {datetime.now().isoformat()}", ""]

    for condition, payload in results.get("conditions", {}).items():
        runs = payload.get("runs", [])
        lines.append(f"[Condição] {condition}")
        lines.append(f"Descrição: {payload.get('description', 'N/D')}")

        overall_scores = []
        gsm8k_scores = []
        bbh_scores = []
        inf_times = []

        for run in runs:
            ev = run.get("evaluation", {})
            overall_scores.append(ev.get("overall_score", 0.0))
            gsm8k_scores.append(ev.get("gsm8k", {}).get("accuracy", 0.0))
            bbh_scores.append(ev.get("bbh", {}).get("average_accuracy", 0.0))
            inf_times.append(ev.get("efficiency", {}).get("inference_speed_seconds", 0.0))

        def _fmt_mean(arr: Sequence[float]) -> str:
            return f"{float(np.mean(arr)):.4f}" if arr else "N/D"

        lines.append(f"Seeds: {[r.get('seed') for r in runs]}")
        lines.append(f"GSM8K acc (média): {_fmt_mean(gsm8k_scores)}")
        lines.append(f"BBH acc (média): {_fmt_mean(bbh_scores)}")
        lines.append(f"Overall (média): {_fmt_mean(overall_scores)}")
        lines.append(f"Inference time s (média): {_fmt_mean(inf_times)}")
        lines.append("")

    ht = results.get("hypothesis_testing")
    if ht:
        lines.append("=== Testes de Hipóteses ===")
        lines.append(json.dumps(ht, indent=2, ensure_ascii=False, default=str))

    lines.append("\n=== FIM DO RESUMO ===")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
