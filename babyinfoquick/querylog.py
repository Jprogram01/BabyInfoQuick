from __future__ import annotations

import json
from datetime import UTC, datetime

from babyinfoquick.config import LOGS_DIR, ensure_dirs, settings
from babyinfoquick.retrieval.pipeline import RetrievalResult
from babyinfoquick.types import Answer


def log_query(question: str, result: RetrievalResult, answer: Answer) -> None:
    if not settings.query_logging:
        return
    ensure_dirs()
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "question": question,
        "config": answer.config_name,
        "age_bands": [b.value for b in result.age_bands],
        "age_source": result.age_source,
        "n_dense": result.n_dense,
        "n_lexical": result.n_lexical,
        "n_before_threshold": result.n_before_threshold,
        "retrieval_ms": round(result.retrieval_ms, 2),
        "total_ms": round(answer.total_ms, 2),
        "answered": answer.answered,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "retrieved": [
            {
                "chunk_id": s.chunk.chunk_id,
                "score": round(s.score, 5),
                "dense": round(s.dense_score, 5) if s.dense_score is not None else None,
                "lexical": round(s.lexical_score, 5)
                if s.lexical_score is not None
                else None,
                "rerank": round(s.rerank_score, 5)
                if s.rerank_score is not None
                else None,
                "age_bands": [b.value for b in s.chunk.age_bands],
                "section": s.chunk.section,
            }
            for s in result.results
        ],
    }
    path = LOGS_DIR / f"queries_{datetime.now(UTC):%Y%m%d}.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
