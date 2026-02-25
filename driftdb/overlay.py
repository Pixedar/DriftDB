from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embedder import embed_texts
from .flow import (
    FlowModel,
    assign_to_attractors,
    build_flow_model,
    chapters_from_labels,
    find_attractors,
    simulate,
    stability_score,
)
from .storage import DriftStore, _ts_to_unix_seconds


class DriftOverlay:
    """Use DriftDB directly as a Python library (no HTTP server).

    This class mounts a local Chroma store (by directory path) and exposes DriftDB's
    flow-first operations (flow probes, simulations, attractors, chapters).
    """

    def __init__(self, chroma_path: str = "./driftdb_chroma"):
        self.store = DriftStore(chroma_path)

    def close(self) -> None:
        self.store.close()

    # ---- collections

    def create_collection(self, name: str, dim: int, distance: str = "cosine", embed_model: Optional[str] = None) -> None:
        self.store.create_collection(name=name, dim=dim, distance=distance, embed_model=embed_model)

    # ---- points

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> int:
        info = self.store.get_collection(collection)
        rows: List[Tuple[str, str, np.ndarray, Dict[str, Any], Optional[str]]] = []
        for i, p in enumerate(points):
            pid = str(p.get("id") or f"auto-{i}")
            ts = str(p["timestamp"])
            payload = dict(p.get("payload") or {})
            text = p.get("text")
            vec = p.get("vector")

            if vec is None and text is None:
                raise ValueError("Each point must have 'vector' or 'text'.")

            if vec is None:
                if not info.embed_model:
                    raise ValueError("Collection has no embed_model; provide 'vector'.")
                emb = embed_texts([str(text or "")], info.embed_model)[0]
                v = np.asarray(emb, dtype=np.float32)
            else:
                v = np.asarray(vec, dtype=np.float32)

            if v.size != info.dim:
                raise ValueError(f"Vector dim mismatch for '{pid}': expected {info.dim}, got {v.size}")

            rows.append((pid, ts, v, payload, str(text) if text is not None else None))

        return self.store.upsert_points(collection, rows)

    def trajectory(self, collection: str, time_min: Optional[str] = None, time_max: Optional[str] = None, limit: int = 10000):
        tmin = _ts_to_unix_seconds(time_min) if time_min else None
        tmax = _ts_to_unix_seconds(time_max) if time_max else None
        return self.store.fetch_points(collection, time_min=tmin, time_max=tmax, limit=limit, order="asc")

    # ---- flow

    def _embed_if_needed(self, collection: str, text: Optional[str], vec: Optional[List[float]]) -> np.ndarray:
        info = self.store.get_collection(collection)
        if vec is not None:
            v = np.asarray(vec, dtype=np.float32)
            if v.size != info.dim:
                raise ValueError(f"Vector dim mismatch: expected {info.dim}, got {v.size}")
            return v
        if text is None:
            raise ValueError("Provide either 'vector' or 'text'.")
        if not info.embed_model:
            raise ValueError("Collection has no embed_model; provide 'vector' explicitly.")
        emb = embed_texts([text], info.embed_model)[0]
        if emb.size != info.dim:
            raise ValueError(f"Embedding model produced dim={emb.size} but collection dim={info.dim}.")
        return emb

    def _build_model(self, collection: str, bandwidth: Optional[float], min_dt_seconds: int) -> FlowModel:
        info = self.store.get_collection(collection)
        X, ids, ts, payloads = self.store.get_matrix(collection)
        return build_flow_model(X=X, ts=ts, metric=info.distance, min_dt_seconds=min_dt_seconds, bandwidth=bandwidth)

    def flow_vector(
        self,
        collection: str,
        *,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        bandwidth: Optional[float] = None,
        min_dt_seconds: int = 60,
    ) -> Dict[str, Any]:
        q = self._embed_if_needed(collection, text, vector)
        model = self._build_model(collection, bandwidth=bandwidth, min_dt_seconds=min_dt_seconds)
        v = model.v_at(q)
        return {"vector": v.astype(float).tolist(), "bandwidth": float(model.bandwidth)}

    def flow_simulate(
        self,
        collection: str,
        *,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        steps: int = 120,
        dt_seconds: float = 3600.0,
        bandwidth: Optional[float] = None,
        min_dt_seconds: int = 60,
        clamp_speed: Optional[float] = None,
    ) -> Dict[str, Any]:
        q = self._embed_if_needed(collection, text, vector)
        model = self._build_model(collection, bandwidth=bandwidth, min_dt_seconds=min_dt_seconds)
        path, speeds = simulate(model, x0=q, steps=int(steps), dt_seconds=float(dt_seconds), clamp_speed=clamp_speed)
        return {
            "path": path.astype(float).tolist(),
            "speeds": speeds.astype(float).tolist(),
            "bandwidth": float(model.bandwidth),
        }

    def flow_attractors(
        self,
        collection: str,
        *,
        seeds: int = 64,
        steps: int = 200,
        dt_seconds: float = 3600.0,
        bandwidth: Optional[float] = None,
        min_dt_seconds: int = 60,
    ) -> Dict[str, Any]:
        model = self._build_model(collection, bandwidth=bandwidth, min_dt_seconds=min_dt_seconds)
        if model.X.shape[0] == 0:
            return {"attractors": [], "bandwidth": float(model.bandwidth), "labels": []}

        n = model.X.shape[0]
        k = min(int(seeds), n)
        idx = np.random.choice(n, size=k, replace=False)
        seed_pts = model.X[idx]

        centers, labels = find_attractors(model, seed_pts, steps=int(steps), dt_seconds=float(dt_seconds))
        attractors = []
        for i in range(centers.shape[0]):
            attractors.append(
                {"id": int(i), "center": centers[i].astype(float).tolist(), "stability": float(stability_score(model, centers[i]))}
            )
        return {"attractors": attractors, "bandwidth": float(model.bandwidth), "labels": labels}

    def flow_chapters(
        self,
        collection: str,
        *,
        seeds: int = 64,
        steps: int = 200,
        dt_seconds: float = 3600.0,
        bandwidth: Optional[float] = None,
        min_dt_seconds: int = 60,
    ) -> Dict[str, Any]:
        info = self.store.get_collection(collection)
        X, ids, ts, payloads = self.store.get_matrix(collection)
        model = build_flow_model(X=X, ts=ts, metric=info.distance, min_dt_seconds=min_dt_seconds, bandwidth=bandwidth)
        if X.shape[0] == 0:
            return {"chapters": [], "attractors": []}

        n = model.X.shape[0]
        k = min(int(seeds), n)
        idx = np.random.choice(n, size=k, replace=False)
        seed_pts = model.X[idx]
        attractor_centers, _ = find_attractors(model, seed_pts, steps=int(steps), dt_seconds=float(dt_seconds))

        labels = assign_to_attractors(X, attractor_centers)
        chapters = chapters_from_labels(ids=ids, ts=ts, labels=labels)

        attractors = []
        for i in range(attractor_centers.shape[0]):
            attractors.append(
                {"id": int(i), "center": attractor_centers[i].astype(float).tolist(), "stability": float(stability_score(model, attractor_centers[i]))}
            )

        return {"chapters": chapters, "attractors": attractors}


class DriftDB(DriftOverlay):
    """Alias for :class:`DriftOverlay`.

    The recommended local (no-server) entry point.
    """
