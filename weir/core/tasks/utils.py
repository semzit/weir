from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotate_vector(quat: np.ndarray, vector: tuple[float, float, float]) -> np.ndarray:
    """Rotate a 3-vector by a quaternion in (w, x, y, z) order.

    SciPy takes quaternions in (x, y, z, w) order and normalizes them.
    """
    w, x, y, z = (float(v) for v in quat[0:4])
    rotation = Rotation.from_quat([x, y, z, w])
    return rotation.apply(vector).astype(np.float32)
