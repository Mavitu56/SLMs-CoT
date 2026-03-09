#!/usr/bin/env python3
"""
Entry point: load config → (optional) sanity checks → train → save plots.

Usage
-----
    python scripts/run.py                              # uses default config
    python scripts/run.py --config configs/kd_qwen_gsm8k.yaml
    python scripts/run.py --config configs/kd_qwen_gsm8k.yaml --output-dir results/run_01
    python scripts/run.py --config configs/dolly_fkl_T2_seed42.yaml \
        --drive-root /content/drive/MyDrive/SLM_results
"""

from __future__ import annotations

import argparse
import json
import sys
import os

import yaml

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_manual import load_teacher, load_student, train
from src.sanity import run_all_sanity_checks
from transformers import AutoTokenizer


DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "kd_qwen_gsm8k.yaml")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded from {path}")
    return cfg


def _plot_losses(log_file: str, output_dir: str, kd_mode: str = "fkl") -> None:
    """Read the JSONL log and save loss-curve plots as PNG."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    if not os.path.isfile(log_file):
        print(f"[plot] log file not found, skipping plots: {log_file}")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        print("[plot] log file is empty, skipping plots.")
        return

    steps      = [r["step"]       for r in rows]
    loss_total = [r["loss_total"] for r in rows]
    loss_ce    = [r["loss_ce"]    for r in rows]
    loss_kd    = [r["loss_kd"]    for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # Panel 1 – CE
    ax = axes[0]
    ax.plot(steps, loss_ce, linewidth=1.5)
    ax.set_title("loss_ce (hard-label cross-entropy)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("CE loss")
    ax.grid(True, alpha=0.3)

    # Panel 2 – KD
    ax = axes[1]
    ax.plot(steps, loss_kd, linewidth=1.5)
    kd_label = {"fkl": "forward KL", "rkl": "reverse KL", "ce_only": "N/A"}.get(kd_mode, kd_mode)
    ax.set_title(f"loss_kd ({kd_label})")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("KD loss")
    ax.grid(True, alpha=0.3)

    # Panel 3 – Total
    ax = axes[2]
    ax.plot(steps, loss_total, linewidth=1.5)
    ax.set_title("loss_total (combined)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Total loss")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training loss curves", fontsize=14, y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "loss_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KD training runner")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save logs, checkpoints and plots. "
             "Overrides log_file and save_dir in the config.",
    )
    parser.add_argument(
        "--drive-root", type=str, default=None,
        help="Google Drive root directory (e.g. /content/drive/MyDrive/SLM_results). "
             "Auto-derives run name from config as {kd_mode}_T{temperature}_seed{seed}.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Apply --drive-root (auto-derives --output-dir from config) ----
    if args.drive_root is not None:
        if args.output_dir is not None:
            print("WARNING: --drive-root overrides --output-dir")
        kd_mode = cfg["kd_mode"]
        temperature = int(cfg["temperature"])
        seed = cfg["seed"]
        run_name = f"{kd_mode}_T{temperature}_seed{seed}"
        args.output_dir = os.path.join(args.drive_root, run_name)
        print(f"Drive root: {args.drive_root}")
        print(f"Run name:   {run_name}")

    # ---- Apply --output-dir overrides ----
    if args.output_dir is not None:
        out = args.output_dir
        os.makedirs(out, exist_ok=True)
        cfg["log_file"] = os.path.join(out, "train_log.jsonl")
        cfg["save_dir"] = os.path.join(out, "checkpoints")
        print(f"Output directory: {out}")
        print(f"  log_file  → {cfg['log_file']}")
        print(f"  save_dir  → {cfg['save_dir']}")

    # ---- Optional sanity checks ----
    if cfg.get("run_sanity", False):
        print("\n>>> Running sanity checks …\n")

        tokenizer = AutoTokenizer.from_pretrained(cfg["student_name"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        teacher = load_teacher(cfg["teacher_name"], cfg["teacher_load_mode"])
        student = load_student(cfg["student_name"], cfg["student_dtype"])

        run_all_sanity_checks(teacher, student, tokenizer, cfg)

        # Free sanity models to reclaim memory before training
        del teacher, student, tokenizer
        import torch
        torch.cuda.empty_cache()

        print("\n>>> Sanity checks done – proceeding to training\n")

    # ---- Train ----
    train(cfg)

    # ---- Generate plots from JSONL log ----
    log_file = cfg.get("log_file")
    if log_file:
        out_dir = os.path.dirname(log_file) or "."
        _plot_losses(log_file, out_dir, kd_mode=cfg.get("kd_mode", "fkl"))


if __name__ == "__main__":
    main()
