# DriftDB

**DriftDB** is a **flow-native vector database** for **timestamped embeddings**.

It treats your vectors as a **trajectory through semantic space** and adds dynamical structure on top of storage:
- estimate a smooth **vector field (flow)** from observed transitions
- query **local motion** at any point
- **simulate paths** by following the flow
- find stable regions (**attractors**) and split timelines into **regimes/chapters**

Under the hood, DriftDB uses **Chroma** as its default storage engine — so it can run standalone, or attach to an **existing Chroma directory** as an analytical layer.


## Origin & Use Case

Below is a short demo video showing one possible application: a dataset of **timestamped diary-entry embeddings** visualized as a point cloud. Each point is an entry, and clusters correspond to recurring “states”.

By modeling these entries as a dynamical system, you can map the actual emotional mechanics at play. The video demonstrates the core mathematical engine powering DriftDB: estimating the underlying vector field (flow) of a person's mood, discovering stable emotional regions (attractors), and simulating paths through latent space.


Watch the demo: https://www.youtube.com/watch?v=taH7kT4x86c

While the original app was a specialized UI for emotional journaling, **DriftDB extracts this engine into a general-purpose API.** It allows you to apply this exact same flow analysis, attractor detection, and trajectory simulation to *any* time-series text or embedding data—such as LLM agent memory, user session behavior, or shifting market narratives.

## Install

```bash
pip install driftdb
```

Optional (server-side text → embeddings via `fastembed`):

```bash
pip install "driftdb[embed]"
```

## Run (local)

By default DriftDB uses a local persistent Chroma directory at `./driftdb_chroma`.

To attach DriftDB to an existing Chroma store, point it at that directory via `DRIFTDB_CHROMA_PATH`.

```bash
export DRIFTDB_CHROMA_PATH=./driftdb_chroma
driftdb
```

OpenAPI:
- http://127.0.0.1:8000/docs

## HTTP API

All endpoints are under `/v1`.

### Create a collection

```bash
curl -X POST http://127.0.0.1:8000/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "notes",
    "dim": 384,
    "distance": "cosine",
    "embed_model": "BAAI/bge-small-en-v1.5"
  }'
```

### Upsert timestamped points

Provide either:
- `vector` directly, or
- `text` (requires `driftdb[embed]` and `embed_model` set on the collection)

```bash
curl -X POST http://127.0.0.1:8000/v1/collections/notes/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "d1",
        "timestamp": "2026-02-20T21:10:00Z",
        "text": "Long day. Focused work, ended calmer.",
        "payload": {"tag": "work"}
      },
      {
        "id": "d2",
        "timestamp": "2026-02-21T21:10:00Z",
        "text": "More social. Feeling lighter and curious.",
        "payload": {"tag": "friends"}
      }
    ]
  }'
```

### Trajectory (time slice)

```bash
curl "http://127.0.0.1:8000/v1/collections/notes/trajectory?time_min=2026-02-01T00:00:00Z&time_max=2026-03-01T00:00:00Z"
```

### Flow probe (local velocity)

```bash
curl -X POST http://127.0.0.1:8000/v1/collections/notes/flow/vector \
  -H "Content-Type: application/json" \
  -d '{"text":"a calm focused day"}'
```

### Follow the flow (simulation)

```bash
curl -X POST http://127.0.0.1:8000/v1/collections/notes/flow/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "a calm focused day",
    "steps": 120,
    "dt_seconds": 43200
  }'
```

### Attractors

```bash
curl "http://127.0.0.1:8000/v1/collections/notes/flow/attractors?seeds=64&steps=200"
```

### Chapters / regimes

```bash
curl "http://127.0.0.1:8000/v1/collections/notes/flow/chapters"
```

## Python client

```python
from driftdb.client import DriftClient

c = DriftClient("http://127.0.0.1:8000")

c.create_collection(name="notes", dim=384, distance="cosine", embed_model="BAAI/bge-small-en-v1.5")
c.upsert("notes", points=[{"id":"d1","timestamp":"2026-02-20T21:10:00Z","text":"...", "payload":{}}])

v = c.flow_vector("notes", text="calm focused day")
path = c.flow_simulate("notes", text="calm focused day", steps=80, dt_seconds=43200)
attractors = c.flow_attractors("notes")
chapters = c.flow_chapters("notes")
```


## Python library (no server)

```python
from driftdb import DriftDB

db = DriftDB("./driftdb_chroma")
db.create_collection("notes", dim=384, distance="cosine", embed_model="BAAI/bge-small-en-v1.5")
db.upsert("notes", [{"id":"d1","timestamp":"2026-02-20T21:10:00Z","text":"...","payload":{}}])

print(db.flow_attractors("notes"))
```

