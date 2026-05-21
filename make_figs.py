"""
Generate the 5 figures for the paper from per-run analysis_results.json files.

Usage
-----
    python make_figs.py --results-dir <path> --out-dir figs/

Where <path> is a directory whose subfolders contain the analysis_results.json
produced by `scripts/run_analysis.py`. The expected structure is:

    <results-dir>/
      ce_only_T1_seed42/analysis/analysis_results.json
      fkl_T1_seed42/analysis/analysis_results.json
      fkl_T2_seed42/analysis/analysis_results.json
      fkl_T4_seed42/analysis/analysis_results.json
      rkl_T1_seed42/analysis/analysis_results.json   # used as RKL representative

The script reads only the keys it needs:
    region_reasoning.entropy
    region_answer.entropy
    region_answer.ece
    rho_HR_HA
    gsm8k_accuracy
    format_failure_rate
    ent_per_position
    kl_per_position

Outputs (PDF, 300 dpi):
    ece_entropy_bars.pdf
    entropy_per_position.pdf
    kl_per_position.pdf
    accuracy_format.pdf
    rho_summary.pdf

Style follows Article I: matplotlib defaults, no seaborn, sober palette,
serif font, top/right spines off.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --------------------------------------------------------------------------
# Conditions and run -> directory mapping
# --------------------------------------------------------------------------
CONDS = [
    # (display label, run-directory name, line color, linestyle for curves)
    ("CE",      "ce_only_T1_seed42", "#4a4a4a", "-"),
    ("FKL T=1", "fkl_T1_seed42",     "#3b6fb5", "-"),
    ("FKL T=2", "fkl_T2_seed42",     "#5fa6e0", "--"),
    ("FKL T=4", "fkl_T4_seed42",     "#1e3f6f", ":"),
    ("RKL",     "rkl_T1_seed42",     "#b5793b", "-"),
]


def load_runs(results_dir: Path) -> dict:
    runs = {}
    for label, dirname, _, _ in CONDS:
        p = results_dir / dirname / "analysis" / "analysis_results.json"
        if not p.exists():
            print(f"warning: missing {p}", file=sys.stderr)
            continue
        with open(p) as f:
            d = json.load(f)
        # If the JSON is wrapped in a top-level key (e.g. {"ce_only_T1_seed42": {...}})
        # unwrap it.
        if len(d) == 1 and isinstance(next(iter(d.values())), dict) \
                and "region_answer" in next(iter(d.values())):
            d = next(iter(d.values()))
        runs[label] = d
    return runs


def smooth(x, w=15):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_ece_entropy_bars(runs, out_path):
    labels = [c[0] for c in CONDS]
    ece_a = [runs[L]["region_answer"]["ece"] for L in labels]
    h_R   = [runs[L]["region_reasoning"]["entropy"] for L in labels]
    h_A   = [runs[L]["region_answer"]["entropy"] for L in labels]

    x = np.arange(len(labels))
    width = 0.28
    fig, ax1 = plt.subplots(figsize=(7.0, 3.6))

    b1 = ax1.bar(x - width, ece_a, width, label="ECE (answer phase)",
                 color="#4a4a4a", edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("ECE (answer phase)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(ece_a) * 1.25)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    b2 = ax2.bar(x,         h_R, width, label=r"$H_R$ (reasoning)",
                 color="#3b6fb5", edgecolor="black", linewidth=0.6, alpha=0.85)
    b3 = ax2.bar(x + width, h_A, width, label=r"$H_A$ (answer)",
                 color="#b5793b", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax2.set_ylabel("Mean token-level entropy")
    ax2.set_ylim(0, max(max(h_R), max(h_A)) * 1.25)

    handles = [b1, b2, b3]
    ax1.legend(handles, [h.get_label() for h in handles],
               loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def _per_position_curves(runs, key, ylabel, ymax_clip, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for label, _, color, ls in CONDS:
        if label not in runs: continue
        y = runs[label][key]
        ys = smooth(y, w=15)
        cutoff = min(450, len(ys))
        ax.plot(np.arange(cutoff), ys[:cutoff], color=color, linestyle=ls,
                label=label, linewidth=1.4)
    ax.set_xlabel(r"$t$ (token position in response)")
    ax.set_ylabel(ylabel)
    if ymax_clip is not None:
        ax.set_ylim(top=ymax_clip)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def fig_entropy_per_position(runs, out_path):
    _per_position_curves(
        runs, "ent_per_position",
        ylabel=r"$H(t)$ (smoothed, window 15)",
        ymax_clip=None, out_path=out_path,
    )


def fig_kl_per_position(runs, out_path):
    # KL has a couple of huge spikes near pad/EOS that compress the curve;
    # cap the y-axis to keep the body readable.
    _per_position_curves(
        runs, "kl_per_position",
        ylabel=r"$\mathrm{KL}(p_T \,\Vert\, p_S)$ at position $t$ (smoothed)",
        ymax_clip=2.5, out_path=out_path,
    )


def fig_accuracy_format(runs, out_path):
    labels = [c[0] for c in CONDS]
    acc = [runs[L]["gsm8k_accuracy"] * 100 for L in labels]
    fmt = [runs[L]["format_failure_rate"] * 100 for L in labels]

    x = np.arange(len(labels)); width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    b1 = ax.bar(x - width/2, acc, width, label="GSM8K accuracy (%)",
                color="#3b6fb5", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width/2, fmt, width, label="Format-failure rate (%)",
                color="#b53b3b", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper left")
    for rect, v in zip(b1, acc):
        ax.text(rect.get_x() + rect.get_width()/2, v + 1.5,
                f"{v:.1f}", ha="center", fontsize=8)
    for rect, v in zip(b2, fmt):
        ax.text(rect.get_x() + rect.get_width()/2, v + 1.5,
                f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def fig_rho_summary(runs, out_path):
    labels = [c[0] for c in CONDS]
    rho = [runs[L]["rho_HR_HA"] for L in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(x, rho, 0.55, color="#4a4a4a", edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="#b53b3b", linewidth=0.9, linestyle="--",
               label=r"$\rho = 1$ (phase symmetry)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\rho = H_R / H_A$")
    ax.set_yscale("log"); ax.set_ylim(0.5, 20)
    ax.legend(frameon=False, loc="upper left")
    for rect, v in zip(bars, rho):
        ax.text(rect.get_x() + rect.get_width()/2, v * 1.08,
                f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path,
                   help="Directory containing per-run subfolders with "
                        "analysis/analysis_results.json")
    p.add_argument("--out-dir", default=Path("figs"), type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    if not runs:
        sys.exit("error: no runs found; check --results-dir")

    fig_ece_entropy_bars(runs,    args.out_dir / "ece_entropy_bars.pdf")
    fig_entropy_per_position(runs, args.out_dir / "entropy_per_position.pdf")
    fig_kl_per_position(runs,      args.out_dir / "kl_per_position.pdf")
    fig_accuracy_format(runs,      args.out_dir / "accuracy_format.pdf")
    fig_rho_summary(runs,          args.out_dir / "rho_summary.pdf")
    print(f"wrote 5 PDFs to {args.out_dir}")


if __name__ == "__main__":
    main()