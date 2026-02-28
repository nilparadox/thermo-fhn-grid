import numpy as np
from typing import Literal

Boundary = Literal["periodic", "noflux"]

def laplacian_2d_periodic(X: np.ndarray) -> np.ndarray:
    # 4-neighbor periodic Laplacian
    return (
        np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
        np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1) -
        4.0 * X
    )

def laplacian_2d_noflux(X: np.ndarray) -> np.ndarray:
    # 4-neighbor Neumann (no-flux) using edge replication
    Xp = np.pad(X, ((1,1),(1,1)), mode="edge")
    return (
        Xp[0:-2,1:-1] + Xp[2:,1:-1] + Xp[1:-1,0:-2] + Xp[1:-1,2:] -
        4.0 * Xp[1:-1,1:-1]
    )

def laplacian_2d(X: np.ndarray, boundary: Boundary) -> np.ndarray:
    if boundary == "periodic":
        return laplacian_2d_periodic(X)
    if boundary == "noflux":
        return laplacian_2d_noflux(X)
    raise ValueError(f"Unknown boundary={boundary}")
