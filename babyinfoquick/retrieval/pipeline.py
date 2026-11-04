from __future__ import annotations

import time

from pydantic import BaseModel

from babyinfoquick.age import AgeBand, extract_age_bands
from babyinfoquick.config import PipelineConfig
from babyinfoquick.index.bm25_store import BM25Store
from babyinfoquick.index.embedder import Embedder
from babyinfoquick.index.qdrant_store import QdrantStore
from babyinfoquick.retrieval.rerank import Reranker
from babyinfoquick.types import Scored


class RetrievalResult(BaseModel):
    query: str
    age_bands: list[AgeBand]
    age_source: str
    results: list[Scored]
    n_dense: int = 0
    n_lexical: int = 0
    n_before_threshold: int = 0
    retrieval_ms: float = 0.0


def reciprocal_rank_fusion(
    runs: list[list[Scored]], rrf_k: int
) -> list[Scored]:
    merged: dict[str, Scored] = {}
    fused: dict[str, float] = {}

    for run in runs:
        for rank, item in enumerate(run):
            chunk_id = item.chunk.chunk_id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = item.model_copy(deep=True)
            else:
                if existing.dense_score is None:
                    existing.dense_score = item.dense_score
                if existing.lexical_score is None:
                    existing.lexical_score = item.lexical_score

    for chunk_id, score in fused.items():
        merged[chunk_id].score = score

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


class Retriever:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.embedder = Embedder(config.embedding) if config.retrieval.dense else None
        self.store = (
            QdrantStore(config.collection, config.embedding.dim)
            if config.retrieval.dense
            else None
        )
        self.bm25 = BM25Store.load(config.collection) if config.retrieval.lexical else None
        self.reranker = Reranker(config.rerank) if config.rerank.enabled else None

    def resolve_age_bands(
        self, query: str, age_bands: list[AgeBand] | None
    ) -> tuple[list[AgeBand], str]:
        if self.config.retrieval.age_filter == "off":
            return [], "disabled"
        if age_bands:
            return age_bands, "explicit"
        extracted = extract_age_bands(query)
        if extracted:
            return extracted, "extracted"
        return [], "none"

    def retrieve(
        self, query: str, age_bands: list[AgeBand] | None = None
    ) -> RetrievalResult:
        started = time.perf_counter()
        retrieval = self.config.retrieval
        bands, age_source = self.resolve_age_bands(query, age_bands)
        pre_filter = bands if retrieval.age_filter == "pre" else None

        runs: list[list[Scored]] = []
        dense_hits: list[Scored] = []
        lexical_hits: list[Scored] = []

        if self.store is not None and self.embedder is not None:
            vector = self.embedder.embed_query(query)
            dense_hits = self.store.search(vector, retrieval.dense_top_k, pre_filter)
            runs.append(dense_hits)

        if self.bm25 is not None:
            lexical_hits = self.bm25.search(query, retrieval.lexical_top_k, pre_filter)
            runs.append(lexical_hits)

        if len(runs) > 1 and retrieval.fusion == "rrf":
            candidates = reciprocal_rank_fusion(runs, retrieval.rrf_k)
        elif runs:
            candidates = sorted(
                {item.chunk.chunk_id: item for run in runs for item in run}.values(),
                key=lambda item: item.score,
                reverse=True,
            )
        else:
            candidates = []

        if retrieval.age_filter == "post" and bands:
            allowed = set(bands)
            candidates = [c for c in candidates if allowed.intersection(c.chunk.age_bands)]

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)

        n_before_threshold = len(candidates)
        if retrieval.min_score > 0:
            candidates = [c for c in candidates if c.score >= retrieval.min_score]

        elapsed = (time.perf_counter() - started) * 1000
        return RetrievalResult(
            query=query,
            age_bands=bands,
            age_source=age_source,
            results=candidates[: retrieval.final_top_k],
            n_dense=len(dense_hits),
            n_lexical=len(lexical_hits),
            n_before_threshold=n_before_threshold,
            retrieval_ms=elapsed,
        )
