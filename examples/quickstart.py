from __future__ import annotations

import datetime as dt
import random

from driftdb.client import DriftClient


def iso(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat()


def main() -> None:
    client = DriftClient("http://127.0.0.1:8000")

    # Minimal example without server-side embeddings: insert vectors directly.
    client.create_collection(name="demo", dim=3, distance="cosine")

    base = dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)

    # A small drifting trajectory in 3D
    x = [0.0, 0.0, 0.0]
    points = []
    for i in range(60):
        x = [x[j] + random.uniform(-0.4, 0.4) for j in range(3)]
        points.append(
            {
                "id": f"d{i:03d}",
                "timestamp": iso(base + dt.timedelta(days=i)),
                "vector": x,
                "payload": {"i": i},
            }
        )

    client.upsert("demo", points=points)

    v = client.flow_vector("demo", vector=[0.2, 0.1, -0.1])
    print("flow_vector:", v)

    path = client.flow_simulate("demo", vector=[0.2, 0.1, -0.1], steps=40, dt_seconds=86400)
    print("simulate steps:", len(path["path"]))

    attractors = client.flow_attractors("demo", seeds=16, steps=80, dt_seconds=86400)
    print("attractors:", len(attractors["attractors"]))

    chapters = client.flow_chapters("demo", seeds=16, steps=80, dt_seconds=86400)
    print("chapters:", len(chapters["chapters"]))


if __name__ == "__main__":
    main()
