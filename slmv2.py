"""Colab entrypoint (non-intrusive).

This file is intentionally small: it keeps Colab conveniences optional and delegates
all experimental logic to `run_experiment.py`.

Usage (recommended in Colab):

1) (Optional) install dependencies in a notebook cell:

   !pip -q install -U transformers datasets accelerate peft evaluate bitsandbytes

2) Run the orchestrator:

   !python run_experiment.py --help
   !python run_experiment.py --kd_modes traditional reasoning --seed 42 --seed 43 \
       --train_limit 3000 --max_length 512 \
       --enable_gsm8k_train --disable_bbh_train \
       --eval_gsm8k --eval_bbh --eval_efficiency \
       --use_cot_prompt_eval \
       --use_logits_kd

This module does NOT auto-run experiments on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ColabSetup:
    """Optional Colab-only setup configuration."""

    mount_drive: bool = True
    drive_mount_point: str = "/content/drive"


def maybe_mount_drive(cfg: ColabSetup) -> None:
    if not cfg.mount_drive:
        return

    try:
        from google.colab import drive  # type: ignore

        drive.mount(cfg.drive_mount_point)
        print(f"✅ Drive montado em {cfg.drive_mount_point}")
    except Exception as exc:
        # Keep non-intrusive: core code should run without Drive.
        print(f"(info) Drive mount pulado: {exc}")


def run_from_notebook(
    args: Optional[List[str]] = None,
    *,
    mount_drive: bool = True,
    drive_mount_point: str = "/content/drive",
) -> int:
    """Programmatic entrypoint for notebooks.

    Example:
        import slmv2
        slmv2.run_from_notebook([
            "--kd_modes", "traditional", "reasoning",
            "--seed", "42", "--seed", "43",
            "--enable_gsm8k_train",
            "--disable_bbh_train",
            "--eval_gsm8k", "--eval_bbh",
            "--use_cot_prompt_eval",
            "--use_logits_kd",
        ])
    """

    maybe_mount_drive(ColabSetup(mount_drive=mount_drive, drive_mount_point=drive_mount_point))

    # Delegate to the CLI runner.
    from run_experiment import main as run_main

    return int(run_main(args))


if __name__ == "__main__":
    # When invoked directly, behave like `python run_experiment.py ...`.
    import sys

    raise SystemExit(run_from_notebook(sys.argv[1:]))
