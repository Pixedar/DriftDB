from __future__ import annotations

import datetime as dt

import numpy as np

from driftdb.client import DriftClient


def iso(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat()


def main() -> None:
    client = DriftClient("http://127.0.0.1:8000")

    dim = 3
    client.create_collection(name="toy3d", dim=dim, distance="l2")

    base = dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)
    points = []
    for i in range(200):
        t = base + dt.timedelta(hours=6 * i)
        a = i / 15.0
        vec = np.array([np.cos(a), np.sin(a), 0.01 * i], dtype=np.float32)
        points.append({"id": f"p{i}", "timestamp": iso(t), "vector": vec.tolist(), "payload": {"i": i}})

    client.upsert("toy3d", points)

    print("Flow at [1,0,0]")
    fv = client.flow_vector("toy3d", vector=[1, 0, 0])
    print(fv)

    print("Attractors")
    atts = client.flow_attractors("toy3d", seeds=32, steps=80, dt_seconds=3600 * 6)
    print(atts["attractors"])

    client.close()


if __name__ == "__main__":
    main()
