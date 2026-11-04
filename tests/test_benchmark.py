import json

from babyinfoquick.config import EVAL_DIR, list_configs
from babyinfoquick.evaluation.labels import load_labels
from babyinfoquick.evaluation.metrics import ConfigReport, RetrievalMetrics


def test_recorded_reports_agree_with_ranked_traces():
    folder = EVAL_DIR / "benchmark"
    labels = load_labels()
    reports = [
        ConfigReport.model_validate(row)
        for row in json.loads((folder / "reports.json").read_text(encoding="utf-8"))
    ]
    assert {report.config_name for report in reports} == set(list_configs())
    for report in reports:
        trace = folder / f"{report.config_name}_retrieval.jsonl"
        rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        assert [row["query_id"] for row in rows] == [label.query_id for label in labels]
        for row in rows:
            assert len(set(row["chunk_ids"])) == len(row["chunk_ids"]) == len(row["relevance"])
            assert row["total_relevant"] >= sum(row["relevance"])
        assert (
            RetrievalMetrics.from_runs(
                [row["relevance"] for row in rows], [row["total_relevant"] for row in rows]
            )
            == report.retrieval
        )
