from __future__ import annotations

import math

from pydantic import BaseModel


def recall_at_k(relevance: list[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    for index, hit in enumerate(relevance, start=1):
        if hit:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevance: list[bool], k: int, total_relevant: int | None = None) -> float:
    gains = [1.0 / math.log2(i + 2) for i, hit in enumerate(relevance[:k]) if hit]
    dcg = sum(gains)
    ideal_hits = min(sum(relevance) if total_relevant is None else total_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[index]


class RetrievalMetrics(BaseModel):
    n: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float

    @classmethod
    def from_runs(
        cls, runs: list[list[bool]], relevant_counts: list[int] | None = None
    ) -> RetrievalMetrics:
        if not runs:
            return cls(n=0, recall_at_1=0, recall_at_5=0, recall_at_10=0, mrr=0, ndcg_at_10=0)
        n = len(runs)
        counts = relevant_counts if relevant_counts is not None else map(sum, runs)
        return cls(
            n=n,
            recall_at_1=sum(recall_at_k(r, 1) for r in runs) / n,
            recall_at_5=sum(recall_at_k(r, 5) for r in runs) / n,
            recall_at_10=sum(recall_at_k(r, 10) for r in runs) / n,
            mrr=sum(reciprocal_rank(r) for r in runs) / n,
            ndcg_at_10=sum(
                ndcg_at_k(run, 10, count) for run, count in zip(runs, counts, strict=True)
            )
            / n,
        )


class AnswerMetrics(BaseModel):
    n: int
    answer_rate: float | None
    citation_precision: float | None
    age_band_correctness: float | None
    faithfulness: float | None = None
    answer_accuracy: float | None = None
    refusal_control_n: int = 0
    refusal_control_rate: float | None = None


class OperationalMetrics(BaseModel):
    p50_retrieval_ms: float
    p95_retrieval_ms: float
    p50_total_ms: float | None
    p95_total_ms: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    mean_cost_usd: float | None


class ConfigReport(BaseModel):
    config_name: str
    description: str
    retrieval: RetrievalMetrics
    retrieval_age_sensitive: RetrievalMetrics
    retrieval_age_agnostic: RetrievalMetrics
    answer: AnswerMetrics
    operational: OperationalMetrics
