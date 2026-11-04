from babyinfoquick.age import AgeBand
from babyinfoquick.evaluation.labels import GoldPassage, is_relevant, normalize
from babyinfoquick.evaluation.metrics import (
    RetrievalMetrics,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from babyinfoquick.types import Chunk


def test_recall_at_k():
    assert recall_at_k([False, False, True], 5) == 1.0
    assert recall_at_k([False, False, True], 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, True, False]) == 0.5
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_rewards_higher_rank():
    assert ndcg_at_k([True, False, False], 10) > ndcg_at_k([False, False, True], 10)


def test_ndcg_perfect_is_one():
    assert ndcg_at_k([True], 10) == 1.0


def test_ndcg_accounts_for_relevant_chunks_outside_the_returned_list():
    assert 0 < ndcg_at_k([True], 10, total_relevant=2) < 1


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 5
    assert percentile([], 95) == 0.0


def test_metrics_aggregate():
    metrics = RetrievalMetrics.from_runs([[True], [False, True]])
    assert metrics.n == 2
    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_5 == 1.0
    assert metrics.mrr == 0.75


def _chunk(text: str, doc_id: str = "d", page: int = 4) -> Chunk:
    return Chunk(
        chunk_id="c1",
        doc_id=doc_id,
        text=text,
        page_start=page,
        page_end=page,
        token_estimate=10,
        title="T",
        url="https://example.org",
        publisher="P",
        doc_type="guidance",
        age_bands=[AgeBand.NEWBORN],
    )


def test_snippet_match_ignores_punctuation_and_case():
    gold = [GoldPassage(doc_id="d", snippet="baby's doctor about solid foods")]
    assert is_relevant(_chunk("Talk with your baby’s doctor about solid foods."), gold)


def test_snippet_must_match_same_document():
    gold = [GoldPassage(doc_id="other", snippet="back to sleep")]
    assert not is_relevant(_chunk("back to sleep"), gold)


def test_page_fallback_matches_range():
    gold = [GoldPassage(doc_id="d", snippet="nothing that appears", page=4)]
    assert not is_relevant(_chunk("unrelated text", page=4), gold)
    gold = [GoldPassage(doc_id="d", page=4)]
    assert is_relevant(_chunk("unrelated text", page=4), gold)
    assert not is_relevant(_chunk("unrelated text", page=9), gold)


def test_normalize():
    assert normalize("  Baby's  DOCTOR!  ") == "baby s doctor"
