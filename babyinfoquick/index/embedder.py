from __future__ import annotations

from functools import lru_cache

import numpy as np

from babyinfoquick.config import EmbeddingConfig


@lru_cache(maxsize=4)
def _load_model(model_name: str, revision: str | None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, revision=revision)


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = _load_model(config.model_name, config.revision)

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            [self.config.query_prefix + text],
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )[0]
