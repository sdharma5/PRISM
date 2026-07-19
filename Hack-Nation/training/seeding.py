"""Reproducible seeding across every RNG this project can touch.

Reported variance must come from the data, not from an unseeded shuffle. Torch is
optional and imported lazily so CPU-only environments stay lightweight.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

#: Upper bound for derived seeds — comfortably inside numpy's 32-bit seed space.
_SEED_MODULUS = 2**31 - 1


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed python, numpy and (if importable) torch. Returns what was actually seeded."""
    seed = int(seed) % _SEED_MODULUS
    seeded: dict[str, Any] = {"seed": seed, "python": True, "numpy": True, "torch": False}

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is an optional dependency; absence must never break a run.
        import torch
    except ImportError:
        return seeded

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        seeded["cuda"] = True
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seeded["deterministic"] = True
    seeded["torch"] = True
    return seeded


def make_rng(seed: int) -> np.random.Generator:
    """A fresh, independent generator — preferred over the global numpy state."""
    return np.random.default_rng(int(seed) % _SEED_MODULUS)


def derive_seed(base_seed: int, *tags: str | int) -> int:
    """Deterministically derive a child seed from a base seed and string tags.

    Using a hash rather than ``base + fold`` avoids accidental collisions between
    (seed 1, fold 2) and (seed 2, fold 1), which would silently correlate runs.
    """
    payload = "|".join([str(base_seed), *[str(t) for t in tags]]).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % _SEED_MODULUS


__all__ = ["derive_seed", "make_rng", "seed_everything"]
