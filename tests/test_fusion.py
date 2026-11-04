from babyinfoquick.age import AgeBand
from babyinfoquick.retrieval.pipeline import reciprocal_rank_fusion
from babyinfoquick.types import Chunk, Scored


def _scored(chunk_id: str, score: float, dense: float | None = None, lexical: float | None = None):
    return Scored(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="d",
            text=chunk_id,
            page_start=1,
            page_end=1,
            token_estimate=5,
            title="T",
            url="https://example.org",
            publisher="P",
            doc_type="guidance",
            age_bands=[AgeBand.NEWBORN],
        ),
        score=score,
        dense_score=dense,
        lexical_score=lexical,
    )


def test_rrf_rewards_agreement_across_runs():
    dense = [_scored("a", 0.9, dense=0.9), _scored("b", 0.8, dense=0.8)]
    lexical = [_scored("b", 5.0, lexical=5.0), _scored("c", 4.0, lexical=4.0)]
    fused = reciprocal_rank_fusion([dense, lexical], rrf_k=60)
    assert fused[0].chunk.chunk_id == "b"


def test_rrf_preserves_component_scores():
    dense = [_scored("a", 0.9, dense=0.9)]
    lexical = [_scored("a", 5.0, lexical=5.0)]
    fused = reciprocal_rank_fusion([dense, lexical], rrf_k=60)
    assert len(fused) == 1
    assert fused[0].dense_score == 0.9
    assert fused[0].lexical_score == 5.0


def test_rrf_deduplicates():
    run = [_scored("a", 1.0), _scored("b", 0.5)]
    fused = reciprocal_rank_fusion([run, run], rrf_k=60)
    assert [f.chunk.chunk_id for f in fused] == ["a", "b"]
