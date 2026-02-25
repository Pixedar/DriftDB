from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import pairwise_distance_matrix


@dataclass
class FlowModel:
    """Non-parametric flow model via Gaussian-kernel regression over (x, v)."""

    X: np.ndarray          # (N, d) anchor states
    V: np.ndarray          # (N, d) observed velocities
    metric: str            # 'cosine' or 'l2'
    bandwidth: float       # kernel bandwidth

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.X.shape[0] == 0:
            return np.zeros_like(x)
        d = pairwise_distance_matrix(self.X, x, self.metric)
        h2 = max(self.bandwidth * self.bandwidth, 1e-12)
        if self.metric == "l2":
            w = np.exp(-d / h2)
        else:
            w = np.exp(-(d * d) / h2)
        sw = float(np.sum(w))
        if sw <= 1e-12:
            return np.zeros_like(x)
        return (w[:, None] * self.V).sum(axis=0) / sw


def build_flow_model(
    X: np.ndarray,
    ts: List[int],
    metric: str,
    min_dt_seconds: int = 60,
    bandwidth: Optional[float] = None,
) -> FlowModel:
    """Build a flow model from time-ordered vectors and timestamps.

    Velocity targets are computed as (x_{i+1} - x_i) / dt.
    Points with dt < min_dt_seconds are skipped.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D array (N, d)")
    d = X.shape[1]
    if X.shape[0] < 2:
        return FlowModel(
            X=np.zeros((0, d), dtype=np.float32),
            V=np.zeros((0, d), dtype=np.float32),
            metric=metric,
            bandwidth=float(bandwidth or 1.0),
        )

    anchors = []
    velos = []
    for i in range(X.shape[0] - 1):
        dt = ts[i + 1] - ts[i]
        if dt < min_dt_seconds:
            continue
        v = (X[i + 1] - X[i]) / float(dt)
        anchors.append(X[i])
        velos.append(v)
    if not anchors:
        return FlowModel(
            X=np.zeros((0, d), dtype=np.float32),
            V=np.zeros((0, d), dtype=np.float32),
            metric=metric,
            bandwidth=float(bandwidth or 1.0),
        )

    A = np.stack(anchors, axis=0).astype(np.float32)
    V = np.stack(velos, axis=0).astype(np.float32)

    if bandwidth is None:
        n = A.shape[0]
        if n >= 10:
            idx = np.random.choice(n, size=min(256, n), replace=False)
            sample = A[idx]
        else:
            sample = A
        if sample.shape[0] < 2:
            bw = 1.0
        else:
            q = sample[0]
            dists = pairwise_distance_matrix(sample[1:], q, metric)
            med = float(np.median(dists))
            bw = max(1e-3, (med ** 0.5) if metric == "l2" else med)
        bandwidth = float(bw)

    return FlowModel(X=A, V=V, metric=metric, bandwidth=float(bandwidth))


def simulate(
    model: FlowModel,
    start: np.ndarray,
    steps: int = 200,
    dt_seconds: float = 3600.0,
    clamp_speed: Optional[float] = None,
) -> np.ndarray:
    """Euler integration: x_{t+1} = x_t + dt * v(x_t)."""
    x = np.asarray(start, dtype=np.float32).copy()
    path = [x.copy()]
    for _ in range(int(steps)):
        v = model.predict(x)
        if clamp_speed is not None:
            speed = float(np.linalg.norm(v))
            if speed > clamp_speed and speed > 1e-12:
                v = v * (clamp_speed / speed)
        x = x + float(dt_seconds) * v
        path.append(x.copy())
    return np.stack(path, axis=0).astype(np.float32)


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> List[int]:
    """Tiny DBSCAN implementation for small N (no sklearn dependency)."""
    n = points.shape[0]
    labels = [-1] * n
    visited = [False] * n
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)

    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(d2[i] <= eps * eps)[0].tolist()
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = neighbors[:]
        while seed_set:
            j = seed_set.pop()
            if not visited[j]:
                visited[j] = True
                nbs = np.where(d2[j] <= eps * eps)[0].tolist()
                if len(nbs) >= min_samples:
                    for k in nbs:
                        if k not in seed_set:
                            seed_set.append(k)
            labels[j] = cluster_id
        cluster_id += 1
    return labels


def find_attractors(
    model: FlowModel,
    seeds: np.ndarray,
    steps: int = 200,
    dt_seconds: float = 3600.0,
    eps: Optional[float] = None,
    min_samples: int = 2,
) -> Tuple[np.ndarray, List[int]]:
    """Simulate from seeds, cluster endpoints, return attractor centers + labels per seed."""
    endpoints = []
    for s in seeds:
        path = simulate(model, s, steps=steps, dt_seconds=dt_seconds, clamp_speed=None)
        endpoints.append(path[-1])
    endpoints = np.stack(endpoints, axis=0).astype(np.float32)

    if eps is None:
        if endpoints.shape[0] < 3:
            eps = 1e-2
        else:
            d2 = np.sum((endpoints[:, None, :] - endpoints[None, :, :]) ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)
            nn = np.sqrt(np.min(d2, axis=1))
            eps = float(np.median(nn)) * 1.5
            eps = max(eps, 1e-4)

    labels = _dbscan(endpoints, eps=float(eps), min_samples=int(min_samples))
    labels_arr = np.array(labels, dtype=int)
    attractors = []
    for cid in sorted(set(l for l in labels if l >= 0)):
        pts = endpoints[labels_arr == cid]
        attractors.append(pts.mean(axis=0))
    if not attractors:
        return np.zeros((0, endpoints.shape[1]), dtype=np.float32), labels
    return np.stack(attractors, axis=0).astype(np.float32), labels


def jacobian_fd(model: FlowModel, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    d = x.size
    J = np.zeros((d, d), dtype=np.float32)
    fx = model.predict(x)
    for i in range(d):
        xp = x.copy()
        xp[i] += h
        fp = model.predict(xp)
        J[:, i] = (fp - fx) / h
    return J


def stability_score(model: FlowModel, x: np.ndarray) -> Dict[str, float]:
    J = jacobian_fd(model, x)
    w = np.linalg.eigvals(J.astype(np.float64))
    max_real = float(np.max(np.real(w))) if w.size else 0.0
    trace = float(np.trace(J))
    return {"trace": trace, "max_eig_real": max_real}


def assign_to_attractors(points: np.ndarray, attractors: np.ndarray) -> np.ndarray:
    if attractors.shape[0] == 0 or points.shape[0] == 0:
        return np.full((points.shape[0],), -1, dtype=int)
    d2 = np.sum((points[:, None, :] - attractors[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1).astype(int)


def chapters_from_labels(ts: List[int], labels: np.ndarray) -> List[Dict[str, object]]:
    if labels.size == 0:
        return []
    segments = []
    start = 0
    for i in range(1, labels.size):
        if int(labels[i]) != int(labels[i - 1]):
            segments.append({
                "label": int(labels[start]),
                "time_min_unix": int(ts[start]),
                "time_max_unix": int(ts[i - 1]),
                "count": int(i - start),
            })
            start = i
    segments.append({
        "label": int(labels[start]),
        "time_min_unix": int(ts[start]),
        "time_max_unix": int(ts[-1]),
        "count": int(labels.size - start),
    })
    return segments
