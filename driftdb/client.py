from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class DriftClient:
    """Minimal SDK-style client for DriftDB."""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def create_collection(
        self, name: str, dim: int, distance: str = "cosine", embed_model: Optional[str] = None
    ) -> Dict[str, Any]:
        r = self._client.post(
            f"{self.url}/v1/collections",
            json={"name": name, "dim": dim, "distance": distance, "embed_model": embed_model},
        )
        r.raise_for_status()
        return r.json()

    def list_collections(self) -> List[Dict[str, Any]]:
        r = self._client.get(f"{self.url}/v1/collections")
        r.raise_for_status()
        return r.json()

    def delete_collection(self, name: str) -> Dict[str, Any]:
        r = self._client.delete(f"{self.url}/v1/collections/{name}")
        r.raise_for_status()
        return r.json()

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        r = self._client.post(f"{self.url}/v1/collections/{collection}/points", json={"points": points})
        r.raise_for_status()
        return r.json()

    def query(
        self,
        collection: str,
        *,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        limit: int = 10,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        with_payload: bool = True,
    ) -> Dict[str, Any]:
        r = self._client.post(
            f"{self.url}/v1/collections/{collection}/query",
            json={
                "vector": vector,
                "text": text,
                "limit": limit,
                "time_min": time_min,
                "time_max": time_max,
                "with_payload": with_payload,
            },
        )
        r.raise_for_status()
        return r.json()

    def trajectory(
        self, collection: str, *, time_min: Optional[str] = None, time_max: Optional[str] = None, limit: int = 10000
    ) -> Dict[str, Any]:
        params = {"limit": str(limit)}
        if time_min:
            params["time_min"] = time_min
        if time_max:
            params["time_max"] = time_max
        r = self._client.get(f"{self.url}/v1/collections/{collection}/trajectory", params=params)
        r.raise_for_status()
        return r.json()

    def flow_vector(
        self,
        collection: str,
        *,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        bandwidth: Optional[float] = None,
        min_dt_seconds: int = 60,
    ) -> Dict[str, Any]:
        r = self._client.post(
            f"{self.url}/v1/collections/{collection}/flow/vector",
            json={"vector": vector, "text": text, "bandwidth": bandwidth, "min_dt_seconds": min_dt_seconds},
        )
        r.raise_for_status()
        return r.json()

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
        r = self._client.post(
            f"{self.url}/v1/collections/{collection}/flow/simulate",
            json={
                "vector": vector,
                "text": text,
                "steps": steps,
                "dt_seconds": dt_seconds,
                "bandwidth": bandwidth,
                "min_dt_seconds": min_dt_seconds,
                "clamp_speed": clamp_speed,
            },
        )
        r.raise_for_status()
        return r.json()

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
        params = {
            "seeds": str(seeds),
            "steps": str(steps),
            "dt_seconds": str(dt_seconds),
            "min_dt_seconds": str(min_dt_seconds),
        }
        if bandwidth is not None:
            params["bandwidth"] = str(bandwidth)
        r = self._client.get(f"{self.url}/v1/collections/{collection}/flow/attractors", params=params)
        r.raise_for_status()
        return r.json()

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
        params = {
            "seeds": str(seeds),
            "steps": str(steps),
            "dt_seconds": str(dt_seconds),
            "min_dt_seconds": str(min_dt_seconds),
        }
        if bandwidth is not None:
            params["bandwidth"] = str(bandwidth)
        r = self._client.get(f"{self.url}/v1/collections/{collection}/flow/chapters", params=params)
        r.raise_for_status()
        return r.json()
