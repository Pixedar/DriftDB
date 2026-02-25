from __future__ import annotations

import numpy as np


def pairwise_distance_matrix(X: np.ndarray, q: np.ndarray, metric: str) -> np.ndarray:
    """Return distances from each row of X to q.

    - cosine: returns (1 - cosine_similarity), range ~[0,2]
    - l2: returns squared L2 distance
    """
    if metric == "cosine":
        q = np.asarray(q, dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-12
        Xn = np.linalg.norm(X, axis=1) + 1e-12
        sims = (X @ q) / (Xn * qn)
        return 1.0 - sims
    if metric == "l2":
        d = X - q[None, :]
        return np.sum(d * d, axis=1)
    raise ValueError(f"Unknown metric: {metric}")
