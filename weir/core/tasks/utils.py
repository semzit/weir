from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from weir.core.interfaces import Observation


def root_x(observation: Observation) -> float:
    """Root x position from a freejoint observation layout (qpos||qvel)."""
    return float(observation[0])


def root_height(observation: Observation) -> float:
    """Root height above the floor from a freejoint observation layout."""
    return float(observation[2])


def root_quat(observation: Observation) -> np.ndarray:
    """Root orientation quaternion in (w, x, y, z) order."""
    return np.asarray(observation[3:7], dtype=np.float32)


def root_vx(observation: Observation, nq: int) -> float:
    """Root linear x velocity; qvel starts at index nq in qpos||qvel."""
    return float(observation[nq])


def rotate_vector(quat: np.ndarray, vector: tuple[float, float, float]) -> np.ndarray:
    """Rotate a 3-vector by a quaternion in (w, x, y, z) order.

    SciPy takes quaternions in (x, y, z, w) order and normalizes them.
    """
    w, x, y, z = (float(v) for v in quat[0:4])
    rotation = Rotation.from_quat([x, y, z, w])
    return rotation.apply(vector).astype(np.float32)
