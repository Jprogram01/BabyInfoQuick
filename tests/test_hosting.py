from collections import deque
from unittest.mock import Mock

from fastapi.testclient import TestClient

from babyinfoquick import api, querylog
from babyinfoquick.config import settings


def test_hosted_question_limit_applies_to_both_routes(monkeypatch):
    monkeypatch.setattr(settings, "max_asks_per_minute", 1)
    monkeypatch.setattr(api, "_ask_times", deque())
    ask = Mock(side_effect=api.HTTPException(503, "retriever unavailable"))
    monkeypatch.setattr(api, "ask", ask)
    client = TestClient(api.app)
    assert client.post("/", data={"question": "example"}).status_code == 503
    response = client.post("/api/ask", json={"question": "another example"})
    assert response.status_code == 429
    assert 0 < int(response.headers["retry-after"]) <= 60
    assert ask.call_count == 1


def test_readiness_requires_lexical_indexes(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "BM25_DIR", tmp_path)
    assert TestClient(api.app).get("/ready").status_code == 503


def test_disabled_query_logging_skips_disk(monkeypatch):
    monkeypatch.setattr(settings, "query_logging", False)
    mkdir = Mock()
    monkeypatch.setattr(querylog, "ensure_dirs", mkdir)
    querylog.log_query("example", Mock(), Mock())
    mkdir.assert_not_called()
