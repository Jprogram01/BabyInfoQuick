from __future__ import annotations

from functools import lru_cache

from babyinfoquick.config import RerankConfig
from babyinfoquick.types import Scored


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, revision: str | None, max_length: int):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, revision=revision, max_length=max_length)


class Reranker:
    def __init__(self, config: RerankConfig):
        self.config = config
        self.model = _load_cross_encoder(config.model_name, config.revision, config.max_length)

    def rerank(self, query: str, candidates: list[Scored]) -> list[Scored]:
        if not candidates:
            return []
        pool = candidates[: self.config.top_k]
        pairs = [(query, item.chunk.text) for item in pool]
        scores = self.model.predict(pairs, batch_size=self.config.batch_size)

        for item, score in zip(pool, scores):
            item.rerank_score = float(score)
            item.score = float(score)

        pool.sort(key=lambda item: item.score, reverse=True)
        return pool
