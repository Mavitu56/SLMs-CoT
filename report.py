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
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Falha ao salvar relatório JSON (arquivo vazio ou ausente): {path}")


def write_summary_txt(path: Path, results: Dict[str, Any]) -> None:
    lines = ["=== RESUMO DE RESULTADOS ===", f"Data: {datetime.now().isoformat()}", ""]

    # Small helper for consistency in summaries.
    def _get_metric(ev: Dict[str, Any], key: str) -> float:
        try:
            v = ev.get(key)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    for condition, payload in results.get("conditions", {}).items():
        runs = payload.get("runs", [])
        lines.append(f"[Condição] {condition}")
        lines.append(f"Descrição: {payload.get('description', 'N/D')}")

        overall_scores = []
        primary_scores = []
        gsm8k_scores = []
        bbh_scores = []
        inf_times = []

        for run in runs:
            ev = run.get("evaluation", {})
            overall_scores.append(_get_metric(ev, "overall_score"))
            primary_scores.append(_get_metric(ev, "primary_score"))
            gsm8k_scores.append(ev.get("gsm8k", {}).get("accuracy", 0.0))
            bbh_scores.append(ev.get("bbh", {}).get("average_accuracy", 0.0))
            inf_times.append(ev.get("efficiency", {}).get("inference_speed_seconds", 0.0))

        def _fmt_mean(arr: Sequence[float]) -> str:
            return f"{float(np.mean(arr)):.4f}" if arr else "N/D"

        lines.append(f"Seeds: {[r.get('seed') for r in runs]}")
        lines.append(f"Primary score (média): {_fmt_mean(primary_scores)}")
        lines.append(f"GSM8K acc (média): {_fmt_mean(gsm8k_scores)}")
        lines.append(f"BBH acc (média): {_fmt_mean(bbh_scores)}")
        lines.append(f"Overall (média): {_fmt_mean(overall_scores)}")
        lines.append(f"Inference time s (média): {_fmt_mean(inf_times)}")

        # Per-run view (more informative than only means).
        if runs:
            lines.append("Runs:")
            lines.append("  seed | primary_score | gsm8k_acc | bbh_acc | inference_s")
            for r in runs:
                ev = r.get("evaluation", {})
                seed = r.get("seed")
                p = _get_metric(ev, "primary_score")
                g = float(ev.get("gsm8k", {}).get("accuracy", 0.0) or 0.0)
                b = float(ev.get("bbh", {}).get("average_accuracy", 0.0) or 0.0)
                t = float(ev.get("efficiency", {}).get("inference_speed_seconds", 0.0) or 0.0)
                lines.append(f"  {seed} | {p:.4f} | {g:.4f} | {b:.4f} | {t:.4f}")
        lines.append("")

    ht = results.get("hypothesis_testing")
    if ht:
        lines.append("=== Testes de Hipóteses ===")
        lines.append(json.dumps(ht, indent=2, ensure_ascii=False, default=str))

    lines.append("\n=== FIM DO RESUMO ===")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Falha ao salvar resumo TXT (arquivo vazio ou ausente): {path}")


def write_plots(out_dir: Path, results: Dict[str, Any], *, prefix: str = "plots") -> List[Path]:
    """Generate simple, automatic visualizations.

    Restriction-friendly: uses only metrics already present in `results`.
    Produces PNGs that can be viewed directly in Colab/Drive.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib indisponível; pulando plots. Erro: {e}")
        return []

    conditions = list((results.get("conditions") or {}).keys())
    if not conditions:
        return []

    # Collect seeds across conditions.
    seed_set = set()
    for cond in conditions:
        for run in (results.get("conditions", {}).get(cond, {}) or {}).get("runs", []) or []:
            try:
                seed_set.add(int(run.get("seed")))
            except Exception:
                pass
    seeds = sorted(seed_set)
    if not seeds:
        return []

    def _extract(cond: str, seed: int, metric: str) -> Optional[float]:
        runs = (results.get("conditions", {}).get(cond, {}) or {}).get("runs", []) or []
        for r in runs:
            if int(r.get("seed")) != int(seed):
                continue
            ev = r.get("evaluation", {}) or {}
            if metric == "gsm8k_accuracy":
                return float(ev.get("gsm8k", {}).get("accuracy", 0.0) or 0.0)
            if metric == "bbh_accuracy":
                return float(ev.get("bbh", {}).get("average_accuracy", 0.0) or 0.0)
            if metric == "primary_score":
                return float(ev.get("primary_score", 0.0) or 0.0)
            if metric == "overall_score":
                return float(ev.get("overall_score", 0.0) or 0.0)
            if metric == "inference_speed_seconds":
                return float(ev.get("efficiency", {}).get("inference_speed_seconds", 0.0) or 0.0)
        return None

    def _grouped_bar(metric: str, title: str, ylabel: str, filename: str) -> Optional[Path]:
        import numpy as _np

        x = _np.arange(len(seeds))
        width = 0.8 / max(1, len(conditions))
        fig, ax = plt.subplots(figsize=(10, 4))
        any_data = False
        for i, cond in enumerate(conditions):
            vals = []
            for s in seeds:
                v = _extract(cond, s, metric)
                vals.append(_np.nan if v is None else float(v))
            if not _np.all(_np.isnan(vals)):
                any_data = True
            ax.bar(x + (i - (len(conditions) - 1) / 2) * width, vals, width, label=cond)

        if not any_data:
            plt.close(fig)
            return None

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("seed")
        ax.legend(loc="best")
        fig.tight_layout()
        out = out_dir / f"{prefix}_{filename}"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        return out

    def _scatter_score_vs_time(filename: str) -> Optional[Path]:
        xs = []
        ys = []
        labels = []
        for cond in conditions:
            for s in seeds:
                score = _extract(cond, s, "primary_score")
                t = _extract(cond, s, "inference_speed_seconds")
                if score is None or t is None:
                    continue
                xs.append(float(t))
                ys.append(float(score))
                labels.append(f"{cond}|{s}")
        if not xs:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=7, alpha=0.8)
        ax.set_title("Primary score vs inference time")
        ax.set_xlabel("inference_speed_seconds")
        ax.set_ylabel("primary_score")
        fig.tight_layout()
        out = out_dir / f"{prefix}_{filename}"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        return out

    outputs: List[Path] = []
    for spec in (
        ("primary_score", "Primary score por seed", "primary_score", "primary_by_seed.png"),
        ("gsm8k_accuracy", "GSM8K accuracy por seed", "accuracy", "gsm8k_by_seed.png"),
        ("bbh_accuracy", "BBH average accuracy por seed", "accuracy", "bbh_by_seed.png"),
        ("inference_speed_seconds", "Inference time (s) por seed", "seconds", "inference_time_by_seed.png"),
    ):
        out = _grouped_bar(*spec)
        if out is not None:
            outputs.append(out)

    out = _scatter_score_vs_time("primary_vs_inference_time.png")
    if out is not None:
        outputs.append(out)

    return outputs
