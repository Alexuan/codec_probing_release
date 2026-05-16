"""Linear Centered Kernel Alignment (paper Sec 2.5).

For two feature matrices ``X`` and ``Y`` shaped ``(N, d_x)`` and ``(N, d_y)``,
linear CKA is::

    CKA(X, Y) = || X^T Y ||_F^2 / ( || X^T X ||_F * || Y^T Y ||_F )

after column-centering both matrices. The random-permutation baseline shuffles
the row alignment of ``Y`` to estimate chance similarity.
"""

from typing import Optional

import numpy as np


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two ``(N, d)`` feature matrices."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row mismatch: {x.shape[0]} vs {y.shape[0]}")

    x = _center(x.astype(np.float64))
    y = _center(y.astype(np.float64))

    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    self_x = np.linalg.norm(x.T @ x, ord="fro")
    self_y = np.linalg.norm(y.T @ y, ord="fro")

    denom = self_x * self_y
    if denom == 0:
        return 0.0
    return float(cross / denom)


def cka_with_baseline(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 100,
    seed: Optional[int] = 0,
) -> dict:
    """CKA plus a random-permutation baseline (paper Sec 2.5)."""
    rng = np.random.default_rng(seed)
    base = linear_cka(x, y)
    perms = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(y.shape[0])
        perms[i] = linear_cka(x, y[perm])
    return {
        "cka": base,
        "baseline_mean": float(perms.mean()),
        "baseline_std": float(perms.std()),
        "delta": float(base - perms.mean()),
        "baseline_samples": perms,
    }
