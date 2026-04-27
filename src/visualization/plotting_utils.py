"""
Plotting utilities for Layer 2 probabilistic analysis.

Generates publication-quality figures comparing CE / KD / KD+CE
across the metrics collected by ``evaluate_probabilistic.py``.

All plots use the Agg (non-interactive) matplotlib backend so they
work headlessly on remote machines.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- Styling defaults ----
_COLORS = {
    "student_base": "#888888",
    "CE": "#1f77b4",
    "KD": "#ff7f0e",
    "KD+CE": "#2ca02c",
}

_MARKERS = {
    "student_base": "s",
    "CE": "o",
    "KD": "^",
    "KD+CE": "D",
}


def _get_color(label: str) -> str:
    return _COLORS.get(label, "#333333")


def _get_marker(label: str) -> str:
    return _MARKERS.get(label, "o")


# ==================================================================
# 1.  Bar chart of scalar metrics
# ==================================================================

def plot_scalar_comparison(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    filename: str = "scalar_metrics.png",
) -> str:
    """Bar chart comparing mean_entropy, mean_maxprob, ece, mean_kl.

    Parameters
    ----------
    results : dict  {label → evaluation result dict}
        e.g. {"CE": {...}, "KD": {...}, "KD+CE": {...}}
    output_dir : str
    filename : str

    Returns
    -------
    Path to the saved figure.
    """
    labels = list(results.keys())
    metrics = ["mean_entropy", "mean_maxprob", "mean_nll", "ppl", "ece"]
    # Include KL only if at least one result has it
    if any(r.get("mean_kl") is not None for r in results.values()):
        metrics.append("mean_kl")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    metric_titles = {
        "mean_entropy": "Mean Entropy (H)",
        "mean_maxprob": "Mean MaxProb",
        "mean_nll": "Mean NLL",
        "ppl": "Perplexity",
        "ece": "ECE",
        "mean_kl": "Mean KL(T‖S)",
    }

    for ax, metric in zip(axes, metrics):
        vals = []
        colors = []
        for lbl in labels:
            v = results[lbl].get(metric)
            vals.append(v if v is not None else 0.0)
            colors.append(_get_color(lbl))

        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(metric_titles.get(metric, metric), fontsize=12)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

        # Annotate values
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.4f}",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("Probabilistic Metrics Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")
    return path


# ==================================================================
# 2.  Per-position KL curve
# ==================================================================

def plot_kl_per_position(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    filename: str = "kl_per_position.png",
    smooth_window: int = 5,
) -> Optional[str]:
    """Line plot of KL(T‖S) vs sequence position for each model.

    Parameters
    ----------
    results : dict {label → eval result dict}
    smooth_window : int
        Simple moving-average window for visual smoothing (1 = no smoothing).

    Returns path or None if no KL data.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    has_data = False

    for lbl, res in results.items():
        kl_curve = res.get("kl_per_position")
        if kl_curve is None or len(kl_curve) == 0:
            continue
        has_data = True
        positions = list(range(len(kl_curve)))
        values = _smooth(kl_curve, smooth_window)
        ax.plot(
            positions, values,
            label=lbl,
            color=_get_color(lbl),
            linewidth=1.2, alpha=0.85,
        )

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Token position (after shift)")
    ax.set_ylabel("KL(Teacher ‖ Student)")
    ax.set_title("KL Divergence per Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")
    return path


# ==================================================================
# 3.  Per-position entropy & maxprob curves
# ==================================================================

def plot_sequence_curves(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    filename: str = "sequence_curves.png",
    smooth_window: int = 5,
) -> str:
    """Two-panel plot: entropy and maxprob vs position.

    Returns path to saved figure.
    """
    fig, (ax_ent, ax_mp) = plt.subplots(1, 2, figsize=(16, 5))

    for lbl, res in results.items():
        # Entropy curve
        ent_curve = res.get("ent_per_position", [])
        if ent_curve:
            positions = list(range(len(ent_curve)))
            ax_ent.plot(
                positions, _smooth(ent_curve, smooth_window),
                label=lbl, color=_get_color(lbl),
                linewidth=1.2, alpha=0.85,
            )

        # MaxProb curve
        mp_curve = res.get("mp_per_position", [])
        if mp_curve:
            positions = list(range(len(mp_curve)))
            ax_mp.plot(
                positions, _smooth(mp_curve, smooth_window),
                label=lbl, color=_get_color(lbl),
                linewidth=1.2, alpha=0.85,
            )

    ax_ent.set_xlabel("Token position")
    ax_ent.set_ylabel("Entropy")
    ax_ent.set_title("Entropy per Position")
    ax_ent.legend()
    ax_ent.grid(True, alpha=0.3)

    ax_mp.set_xlabel("Token position")
    ax_mp.set_ylabel("Max Probability")
    ax_mp.set_title("Max Probability per Position")
    ax_mp.legend()
    ax_mp.grid(True, alpha=0.3)

    fig.suptitle("Sequence-level Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")
    return path


# ==================================================================
# 4.  Region-comparison grouped bar chart
# ==================================================================

def plot_region_comparison(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    filename: str = "region_comparison.png",
) -> Optional[str]:
    """Grouped bar chart: per-region entropy, NLL, maxprob for each model.

    Each panel shows one metric.  Within each panel, bars are grouped
    by model and coloured by region (prompt / reasoning / answer).

    Returns path or None if no region data.
    """
    import numpy as np

    region_keys = [
        ("region_prompt",    "Prompt"),
        ("region_reasoning", "Reasoning"),
        ("region_answer",    "Answer"),
    ]
    region_colors = {
        "Prompt":    "#a0a0a0",
        "Reasoning": "#4c72b0",
        "Answer":    "#dd8452",
    }
    metrics = [
        ("entropy",  "Entropy"),
        ("nll",      "NLL"),
        ("maxprob",  "MaxProb"),
        ("ppl",      "Perplexity"),
    ]

    # Check if any result has region data
    has_data = any(
        res.get(rk, {}).get("n_tokens", 0) > 0
        for res in results.values()
        for rk, _ in region_keys
    )
    if not has_data:
        return None

    model_labels = list(results.keys())
    n_models = len(model_labels)
    n_regions = len(region_keys)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(n_models)
    bar_width = 0.25

    for ax, (mkey, mtitle) in zip(axes, metrics):
        for j, (rk, rname) in enumerate(region_keys):
            vals = []
            for mlbl in model_labels:
                rd = results[mlbl].get(rk, {})
                v = rd.get(mkey)
                vals.append(v if v is not None else 0.0)
            offset = (j - (n_regions - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, vals, bar_width,
                label=rname, color=region_colors[rname],
                edgecolor="black", linewidth=0.4,
            )
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_title(mtitle, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Region Metric Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")
    return path


# ==================================================================
# 5.  Combined summary figure
# ==================================================================

def plot_all(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    smooth_window: int = 5,
) -> List[str]:
    """Generate all standard analysis plots.

    Returns list of saved file paths.
    """
    paths: List[str] = []
    paths.append(plot_scalar_comparison(results, output_dir))
    kl_path = plot_kl_per_position(results, output_dir, smooth_window=smooth_window)
    if kl_path is not None:
        paths.append(kl_path)
    paths.append(plot_sequence_curves(results, output_dir, smooth_window=smooth_window))
    region_path = plot_region_comparison(results, output_dir)
    if region_path is not None:
        paths.append(region_path)
    return paths


# ==================================================================
# Smoothing helper
# ==================================================================

def _smooth(values: list, window: int) -> list:
    """Simple moving average for visual smoothing.

    Window=1 returns original values.
    """
    if window <= 1 or len(values) <= window:
        return values
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed
