from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from dateutil import parser as dtparser


def _ts_to_unix_seconds(ts: str) -> int:
    d = dtparser.isoparse(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def _unix_to_iso(ts_unix: int) -> str:
    return dt.datetime.fromtimestamp(int(ts_unix), tz=dt.timezone.utc).isoformat()


@dataclass(frozen=True)
class CollectionInfo:
    name: str
    dim: int
    distance: str
    embed_model: Optional[str]


@dataclass(frozen=True)
class Point:
    id: str
    timestamp: str
    ts_unix: int
    vector: np.ndarray
    payload: Dict[str, Any]
    text: Optional[str] = None


class DriftStore:
    """Storage backed by a local persistent ChromaDB collection.

    Notes:
    - timestamps are stored as unix seconds in metadata under `_driftdb_ts`
    - arbitrary payload is JSON-encoded into `_driftdb_payload` (Chroma metadata values are scalars)
    """

    def __init__(self, chroma_path: str):
        import chromadb  # dependency in base install

        self.path = chroma_path
        self._client = chromadb.PersistentClient(path=chroma_path)

    def close(self) -> None:
        # PersistentClient has no explicit close in most versions; keep as no-op.
        return None

    # ---- collections

    def create_collection(self, name: str, dim: int, distance: str, embed_model: Optional[str]) -> None:
        if distance not in {"cosine", "l2"}:
            raise ValueError("distance must be 'cosine' or 'l2'")

        # Chroma uses `hnsw:space` for similarity; l2 maps to `l2`, cosine maps to `cosine`.
        hnsw_space = "cosine" if distance == "cosine" else "l2"

        meta = {
            "hnsw:space": hnsw_space,
            "_driftdb_dim": int(dim),
            "_driftdb_distance": distance,
            "_driftdb_embed_model": embed_model or "",
        }

        # If it exists, verify metadata is compatible.
        try:
            existing = self._client.get_collection(name)
            em = getattr(existing, "metadata", None) or {}
            ex_dim = int(em.get("_driftdb_dim", dim))
            ex_dist = em.get("_driftdb_distance", distance)
            if ex_dim != int(dim) or ex_dist != distance:
                raise ValueError(
                    f"Collection '{name}' already exists with dim={ex_dim}, distance={ex_dist}"
                )
            # keep existing; do not overwrite metadata
            return
        except Exception:
            pass

        self._client.get_or_create_collection(name=name, metadata=meta)

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception:
            # delete should be idempotent
            return

    def list_collections(self) -> List[CollectionInfo]:
        out: List[CollectionInfo] = []
        try:
            cols = self._client.list_collections()
        except Exception:
            cols = []
        for c in cols:
            meta = getattr(c, "metadata", None) or {}
            out.append(
                CollectionInfo(
                    name=c.name,
                    dim=int(meta.get("_driftdb_dim", 0) or 0),
                    distance=str(meta.get("_driftdb_distance", "cosine")),
                    embed_model=(meta.get("_driftdb_embed_model") or None) or None,
                )
            )
        return out

    def get_collection(self, name: str) -> CollectionInfo:
        try:
            c = self._client.get_collection(name)
        except Exception:
            raise KeyError(f"collection not found: {name}")
        meta = getattr(c, "metadata", None) or {}
        dim = int(meta.get("_driftdb_dim", 0) or 0)
        distance = str(meta.get("_driftdb_distance", "cosine"))
        embed_model = (meta.get("_driftdb_embed_model") or None) or None
        if dim <= 0:
            raise KeyError(f"collection metadata missing dim: {name}")
        return CollectionInfo(name=name, dim=dim, distance=distance, embed_model=embed_model)

    def _coll(self, name: str):
        # get_collection throws a nice error if missing
        self.get_collection(name)
        return self._client.get_collection(name)

    # ---- points

    def upsert_points(self, collection: str, rows: Iterable[Tuple[str, str, np.ndarray, Dict[str, Any], Optional[str]]]) -> int:
        info = self.get_collection(collection)
        coll = self._coll(collection)

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for pid, timestamp, vec, payload, text in rows:
            v = np.asarray(vec, dtype=np.float32)
            if v.size != info.dim:
                raise ValueError(f"Vector dim mismatch for '{pid}': expected {info.dim}, got {v.size}")
            ts_unix = _ts_to_unix_seconds(timestamp)

            ids.append(str(pid))
            embeddings.append(v.astype(float).tolist())
            metadatas.append(
                {
                    "_driftdb_ts": int(ts_unix),
                    "_driftdb_timestamp": str(timestamp),
                    "_driftdb_payload": json.dumps(payload or {}, ensure_ascii=False),
                }
            )
            documents.append(text or "")

        if not ids:
            return 0

        # Chroma versions differ: prefer `upsert`, fallback to delete+add.
        if hasattr(coll, "upsert"):
            coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        else:
            try:
                coll.delete(ids=ids)
            except Exception:
                pass
            coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

        return len(ids)

    def _get_all(self, coll, include: List[str], batch_size: int = 1024) -> Dict[str, Any]:
        # Fetch everything with pagination; Chroma stores ids/embeddings/metadatas parallel.
        n = int(coll.count())
        all_ids: List[str] = []
        all_embeddings: List[List[float]] = []
        all_metadatas: List[Dict[str, Any]] = []
        all_documents: List[str] = []

        for offset in range(0, n, batch_size):
            res = coll.get(limit=batch_size, offset=offset, include=include)
            all_ids.extend(res.get("ids", []) or [])
            if "embeddings" in include:
                all_embeddings.extend(res.get("embeddings", []) or [])
            if "metadatas" in include:
                all_metadatas.extend(res.get("metadatas", []) or [])
            if "documents" in include:
                all_documents.extend(res.get("documents", []) or [])

        out: Dict[str, Any] = {"ids": all_ids}
        if "embeddings" in include:
            out["embeddings"] = all_embeddings
        if "metadatas" in include:
            out["metadatas"] = all_metadatas
        if "documents" in include:
            out["documents"] = all_documents
        return out

    def get_matrix(
        self,
        collection: str,
        time_min: Optional[int] = None,
        time_max: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, List[Dict[str, Any]]]:
        """Return (X, ids, ts_unix, payloads) sorted by time."""
        coll = self._coll(collection)

        res = self._get_all(coll, include=["embeddings", "metadatas"], batch_size=2048)
        ids = res["ids"]
        embs = res.get("embeddings", [])
        metas = res.get("metadatas", []) or [{} for _ in ids]

        rows = []
        for pid, emb, meta in zip(ids, embs, metas):
            ts = int((meta or {}).get("_driftdb_ts", 0) or 0)
            if time_min is not None and ts < int(time_min):
                continue
            if time_max is not None and ts > int(time_max):
                continue
            payload_json = (meta or {}).get("_driftdb_payload", "{}")
            try:
                payload = json.loads(payload_json) if payload_json else {}
            except Exception:
                payload = {}
            rows.append((ts, str(pid), emb, payload))

        rows.sort(key=lambda r: r[0])

        if not rows:
            return (
                np.zeros((0, self.get_collection(collection).dim), dtype=np.float32),
                [],
                np.zeros((0,), dtype=np.int64),
                [],
            )

        ts = np.asarray([r[0] for r in rows], dtype=np.int64)
        ids_out = [r[1] for r in rows]
        X = np.asarray([r[2] for r in rows], dtype=np.float32)
        payloads = [r[3] for r in rows]
        return X, ids_out, ts, payloads

    def fetch_points(
        self,
        collection: str,
        time_min: Optional[int] = None,
        time_max: Optional[int] = None,
        limit: int = 10000,
        order: str = "asc",
    ) -> List[Point]:
        """Fetch points as Point objects (sorted by time)."""
        coll = self._coll(collection)
        res = self._get_all(coll, include=["embeddings", "metadatas", "documents"], batch_size=2048)

        rows = []
        for pid, emb, meta, doc in zip(
            res.get("ids", []),
            res.get("embeddings", []),
            res.get("metadatas", []) or [],
            res.get("documents", []) or [],
        ):
            ts = int((meta or {}).get("_driftdb_ts", 0) or 0)
            if time_min is not None and ts < int(time_min):
                continue
            if time_max is not None and ts > int(time_max):
                continue
            rows.append((ts, pid, emb, meta, doc))

        reverse = order.lower() == "desc"
        rows.sort(key=lambda r: r[0], reverse=reverse)
        rows = rows[: int(limit)]

        out: List[Point] = []
        for ts, pid, emb, meta, doc in rows:
            payload_json = (meta or {}).get("_driftdb_payload", "{}")
            try:
                payload = json.loads(payload_json) if payload_json else {}
            except Exception:
                payload = {}
            timestamp = (meta or {}).get("_driftdb_timestamp") or _unix_to_iso(ts)
            out.append(
                Point(
                    id=str(pid),
                    timestamp=str(timestamp),
                    ts_unix=int(ts),
                    vector=np.asarray(emb, dtype=np.float32),
                    payload=payload,
                    text=str(doc) if doc is not None else None,
                )
            )
        return out

    def query(
        self,
        collection: str,
        query_vector: np.ndarray,
        limit: int = 10,
        time_min: Optional[int] = None,
        time_max: Optional[int] = None,
        with_payload: bool = True,
    ) -> List[Tuple[str, float, int, Optional[Dict[str, Any]]]]:
        """Return [(id, score, ts_unix, payload_or_none), ...]."""
        info = self.get_collection(collection)
        coll = self._coll(collection)

        q = np.asarray(query_vector, dtype=np.float32)
        if q.size != info.dim:
            raise ValueError(f"Query dim mismatch: expected {info.dim}, got {q.size}")

        # oversample a bit to accommodate optional time filtering
        n_results = max(int(limit), int(limit) * 5)

        res = coll.query(
            query_embeddings=[q.astype(float).tolist()],
            n_results=n_results,
            include=["distances", "metadatas"],
        )

        ids = (res.get("ids") or [[]])[0] or []
        dists = (res.get("distances") or [[]])[0] or [0.0 for _ in ids]
        metas = (res.get("metadatas") or [[]])[0] or [{} for _ in ids]

        out: List[Tuple[str, float, int, Optional[Dict[str, Any]]]] = []
        for pid, dist, meta in zip(ids, dists, metas):
            ts = int((meta or {}).get("_driftdb_ts", 0) or 0)
            if time_min is not None and ts < int(time_min):
                continue
            if time_max is not None and ts > int(time_max):
                continue

            if info.distance == "cosine":
                score = 1.0 - float(dist)
            else:
                score = -float(dist)

            payload = None
            if with_payload:
                payload_json = (meta or {}).get("_driftdb_payload", "{}")
                try:
                    payload = json.loads(payload_json) if payload_json else {}
                except Exception:
                    payload = {}

            out.append((str(pid), score, ts, payload))
            if len(out) >= int(limit):
                break
        return out
