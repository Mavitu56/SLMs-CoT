"""Deterministic seeding for reproducibility."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seed for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuDNN (may hurt perf; acceptable for audit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
