from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np


@lru_cache(maxsize=8)
def _load_fastembed(model_name: str):
    from fastembed import TextEmbedding  # optional dependency

    return TextEmbedding(model_name)


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """Embed texts using fastembed (optional dependency)."""
    model = _load_fastembed(model_name)
    embs = []
    for e in model.embed(texts):
        embs.append(np.asarray(e, dtype=np.float32))
    if not embs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(embs, axis=0).astype(np.float32)
