from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces

from weir.core.interfaces import Shape

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def resolve_model_asset(name: str) -> str:
    """Resolve a bare model file name (or relative path) under the models dir."""
    return str((MODELS_DIR / name).resolve())


def shape_to_box(shape: Shape) -> spaces.Box:
    """Build a gymnasium Box space from a Shape (unbounded when low/high are absent)."""
    low = shape.low if shape.low is not None else -np.inf
    high = shape.high if shape.high is not None else np.inf
    return spaces.Box(low=low, high=high, shape=shape.dims, dtype=np.float32)
