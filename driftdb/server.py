from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from dateutil import parser as dtparser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
from .storage import DriftStore


Distance = Literal["cosine", "l2"]


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1)
    dim: int = Field(..., ge=1)
    distance: Distance = "cosine"
    embed_model: Optional[str] = None


class CollectionResponse(BaseModel):
    name: str
    dim: int
    distance: Distance
    embed_model: Optional[str] = None


class PointIn(BaseModel):
    id: Optional[str] = None
    timestamp: str
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    points: List[PointIn]


class UpsertResponse(BaseModel):
    upserted: int


class QueryRequest(BaseModel):
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    limit: int = 10
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    with_payload: bool = True


class Hit(BaseModel):
    id: str
    score: float
    timestamp: str
    payload: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    points: List[Hit]


class TrajectoryPoint(BaseModel):
    id: str
    timestamp: str
    vector: List[float]
    payload: Dict[str, Any]


class TrajectoryResponse(BaseModel):
    points: List[TrajectoryPoint]


class FlowVectorRequest(BaseModel):
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    bandwidth: Optional[float] = None
    min_dt_seconds: int = 60


class FlowVectorResponse(BaseModel):
    vector: List[float]        # velocity
    speed: float
    bandwidth: float
    samples: int


class SimulateRequest(BaseModel):
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    steps: int = 120
    dt_seconds: float = 3600.0
    bandwidth: Optional[float] = None
    min_dt_seconds: int = 60
    clamp_speed: Optional[float] = None


class SimulateResponse(BaseModel):
    path: List[List[float]]
    speeds: List[float]
    bandwidth: float


class Attractor(BaseModel):
    id: int
    center: List[float]
    stability: Dict[str, float]


class AttractorResponse(BaseModel):
    attractors: List[Attractor]
    bandwidth: float
    labels: List[int]  # per-seed label


class ChaptersResponse(BaseModel):
    chapters: List[Dict[str, Any]]
    attractors: List[Attractor]


def _iso_to_unix(ts: str) -> int:
    dt = dtparser.isoparse(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=dtparser.tz.UTC)
    return int(dt.timestamp())


def _embed_if_needed(store: DriftStore, collection: str, text: Optional[str], vec: Optional[List[float]]) -> np.ndarray:
    info = store.get_collection(collection)
    if vec is not None:
        v = np.asarray(vec, dtype=np.float32)
        if v.size != info.dim:
            raise HTTPException(status_code=400, detail=f"Vector dim mismatch: expected {info.dim}, got {v.size}")
        return v
    if text is None:
        raise HTTPException(status_code=400, detail="Provide either 'vector' or 'text'.")
    if not info.embed_model:
        raise HTTPException(status_code=400, detail="Collection has no embed_model; provide 'vector' explicitly.")
    try:
        emb = embed_texts([text], info.embed_model)[0]
    except ModuleNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="fastembed not installed on server. Install with: pip install driftdb[embed]",
        )
    if emb.size != info.dim:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding model produced dim={emb.size} but collection dim={info.dim}.",
        )
    return emb


def _build_model(store: DriftStore, collection: str, bandwidth: Optional[float], min_dt_seconds: int) -> FlowModel:
    info = store.get_collection(collection)
    X, ids, ts, payloads = store.get_matrix(collection)
    return build_flow_model(X=X, ts=ts, metric=info.distance, min_dt_seconds=min_dt_seconds, bandwidth=bandwidth)


def create_app() -> FastAPI:
    app = FastAPI(
        title="DriftDB",
        description="Flow-native vector time-series database (semantic trajectories, vector fields, attractors).",
        version="0.1.2",
    )

    chroma_path = os.environ.get("DRIFTDB_CHROMA_PATH") or os.environ.get("DRIFTDB_PATH") or "./driftdb_chroma"
    store = DriftStore(chroma_path)

    @app.on_event("shutdown")
    def _shutdown():
        store.close()

    @app.post("/v1/collections", response_model=CollectionResponse)
    def create_collection(req: CreateCollectionRequest):
        try:
            store.create_collection(req.name, req.dim, req.distance, req.embed_model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        info = store.get_collection(req.name)
        return CollectionResponse(**info.__dict__)

    @app.get("/v1/collections", response_model=List[CollectionResponse])
    def list_collections():
        return [CollectionResponse(**c.__dict__) for c in store.list_collections()]

    @app.delete("/v1/collections/{name}")
    def delete_collection(name: str):
        store.delete_collection(name)
        return {"deleted": True}

    @app.post("/v1/collections/{name}/points", response_model=UpsertResponse)
    def upsert_points(name: str, req: UpsertRequest):
        info = store.get_collection(name)

        rows = []
        for i, p in enumerate(req.points):
            pid = p.id or f"auto-{i}"
            if p.vector is None and p.text is None:
                raise HTTPException(status_code=400, detail="Each point must have 'vector' or 'text'.")
            if p.vector is not None:
                v = np.asarray(p.vector, dtype=np.float32)
                if v.size != info.dim:
                    raise HTTPException(status_code=400, detail=f"Vector dim mismatch for '{pid}'.")
            else:
                if not info.embed_model:
                    raise HTTPException(status_code=400, detail="Collection has no embed_model; provide 'vector'.")
                try:
                    v = embed_texts([p.text or ""], info.embed_model)[0]
                except ModuleNotFoundError:
                    raise HTTPException(
                        status_code=500,
                        detail="fastembed not installed on server. Install with: pip install driftdb[embed]",
                    )
                if v.size != info.dim:
                    raise HTTPException(status_code=500, detail="Embedding dim mismatch.")
            rows.append((pid, p.timestamp, v, p.payload, p.text))

        try:
            n = store.upsert_points(name, rows)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return UpsertResponse(upserted=n)

    @app.post("/v1/collections/{name}/query", response_model=QueryResponse)
    def query_points(name: str, req: QueryRequest):
        _ = store.get_collection(name)  # validate
        q = _embed_if_needed(store, name, req.text, req.vector)

        tmin = _iso_to_unix(req.time_min) if req.time_min else None
        tmax = _iso_to_unix(req.time_max) if req.time_max else None

        try:
            hits_raw = store.query(
                name,
                query_vector=q,
                limit=int(req.limit),
                time_min=tmin,
                time_max=tmax,
                with_payload=bool(req.with_payload),
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        import datetime as _dt

        hits = []
        for pid, score, ts_unix, payload in hits_raw:
            timestamp = _dt.datetime.fromtimestamp(int(ts_unix), tz=_dt.timezone.utc).isoformat()
            hits.append(Hit(id=pid, score=float(score), timestamp=timestamp, payload=payload))
        return QueryResponse(points=hits)

    @app.get("/v1/collections/{name}/trajectory", response_model=TrajectoryResponse)
    def trajectory(name: str, time_min: Optional[str] = None, time_max: Optional[str] = None, limit: int = 10000):
        tmin = _iso_to_unix(time_min) if time_min else None
        tmax = _iso_to_unix(time_max) if time_max else None
        pts = store.fetch_points(name, time_min=tmin, time_max=tmax, limit=int(limit), order="asc")
        out = [
            TrajectoryPoint(
                id=p.id,
                timestamp=p.timestamp,
                vector=p.vector.astype(float).tolist(),
                payload=p.payload,
            )
            for p in pts
        ]
        return TrajectoryResponse(points=out)

    @app.post("/v1/collections/{name}/flow/vector", response_model=FlowVectorResponse)
    def flow_vector(name: str, req: FlowVectorRequest):
        model = _build_model(store, name, bandwidth=req.bandwidth, min_dt_seconds=req.min_dt_seconds)
        x = _embed_if_needed(store, name, req.text, req.vector)
        v = model.predict(x)
        return FlowVectorResponse(
            vector=v.astype(float).tolist(),
            speed=float(np.linalg.norm(v)),
            bandwidth=float(model.bandwidth),
            samples=int(model.X.shape[0]),
        )

    @app.post("/v1/collections/{name}/flow/simulate", response_model=SimulateResponse)
    def flow_simulate(name: str, req: SimulateRequest):
        model = _build_model(store, name, bandwidth=req.bandwidth, min_dt_seconds=req.min_dt_seconds)
        x0 = _embed_if_needed(store, name, req.text, req.vector)
        path = simulate(model, x0, steps=req.steps, dt_seconds=req.dt_seconds, clamp_speed=req.clamp_speed)
        speeds = [float(np.linalg.norm(model.predict(path[i]))) for i in range(path.shape[0])]
        return SimulateResponse(
            path=path.astype(float).tolist(),
            speeds=speeds,
            bandwidth=float(model.bandwidth),
        )

    @app.get("/v1/collections/{name}/flow/attractors", response_model=AttractorResponse)
    def flow_attractors(
        name: str,
        seeds: int = 64,
        steps: int = 200,
        dt_seconds: float = 3600.0,
        bandwidth: Optional[float] = None,
        min_dt_seconds: int = 60,
    ):
        model = _build_model(store, name, bandwidth=bandwidth, min_dt_seconds=min_dt_seconds)
        if model.X.shape[0] == 0:
            return AttractorResponse(attractors=[], bandwidth=float(model.bandwidth), labels=[])

        n = model.X.shape[0]
        k = min(int(seeds), n)
        idx = np.random.choice(n, size=k, replace=False)
        seed_pts = model.X[idx]

        attractors, labels = find_attractors(model, seed_pts, steps=int(steps), dt_seconds=float(dt_seconds))
        out = []
        for i in range(attractors.shape[0]):
            st = stability_score(model, attractors[i])
            out.append(Attractor(id=i, center=attractors[i].astype(float).tolist(), stability=st))
        return AttractorResponse(attractors=out, bandwidth=float(model.bandwidth), labels=labels)

    @app.get("/v1/collections/{name}/flow/chapters", response_model=ChaptersResponse)
    def flow_chapters(
        name: str,
        seeds: int = 64,
        steps: int = 200,
        dt_seconds: float = 3600.0,
        bandwidth: Optional[float] = None,
        min_dt_seconds: int = 60,
    ):
        info = store.get_collection(name)
        X, ids, ts, payloads = store.get_matrix(name)
        model = build_flow_model(X=X, ts=ts, metric=info.distance, min_dt_seconds=min_dt_seconds, bandwidth=bandwidth)
        if X.shape[0] == 0:
            return ChaptersResponse(chapters=[], attractors=[])

        n = model.X.shape[0]
        k = min(int(seeds), n)
        idx = np.random.choice(n, size=k, replace=False)
        seed_pts = model.X[idx]
        attractor_centers, _ = find_attractors(model, seed_pts, steps=int(steps), dt_seconds=float(dt_seconds))

        labels = assign_to_attractors(X, attractor_centers)
        chapters = chapters_from_labels(ts, labels)

        atts = []
        for i in range(attractor_centers.shape[0]):
            atts.append(
                Attractor(
                    id=i,
                    center=attractor_centers[i].astype(float).tolist(),
                    stability=stability_score(model, attractor_centers[i]),
                )
            )

        return ChaptersResponse(chapters=chapters, attractors=atts)

    return app


def main() -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
