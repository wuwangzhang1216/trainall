"""Reproducibility helper."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 0, deterministic: bool = False) -> int:
    """Seed python / numpy / torch RNGs.  Returns the seed for convenience."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        pass
    return seed


__all__ = ["seed_everything"]
