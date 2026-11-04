import time
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from babyinfoquick import service
from babyinfoquick.answer.generate import NO_ANSWER_TEXT, _to_answer, generate
from babyinfoquick.answer.llm import GroundedAnswer, Usage
from babyinfoquick.api import app
from babyinfoquick.config import GenerationConfig, load_config, settings
from babyinfoquick.evaluation import runner
from babyinfoquick.evaluation.judge import Verdict, judge_answer
from babyinfoquick.evaluation.labels import GoldPassage, Label
from babyinfoquick.retrieval.pipeline import RetrievalResult
from babyinfoquick.types import Chunk, Scored


@pytest.fixture
def result():
    chunk = Chunk(
        chunk_id="guide:1",
        doc_id="guide",
        text="An example passage.",
        page_start=2,
        page_end=2,
        token_estimate=4,
        title="Guide",
        url="https://example.org/guide.pdf",
        publisher="Example",
        doc_type="guidance",
    )
    return RetrievalResult(
        query="example",
        age_bands=[],
        age_source="none",
        results=[Scored(chunk=chunk, score=1)],
    )


def test_keyless_demo_serves_ranked_passages_in_html_and_json(monkeypatch, result):
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    retriever = Mock(config=load_config("hybrid"))
    retriever.retrieve.return_value = result
    monkeypatch.setattr(service, "get_retriever", lambda name: retriever)
    monkeypatch.setattr(service, "log_query", Mock())
    monkeypatch.setattr(service, "generate", Mock(side_effect=AssertionError))
    client = TestClient(app)
    response = client.post("/api/ask", json={"question": "example", "config": "hybrid"})
    assert response.status_code == 200
    answer = response.json()
    assert not answer["answered"]
    assert "OPENAI_API_KEY or ANTHROPIC_API_KEY" in answer["text"]
    assert answer["retrieved_passages"][0]["chunk"]["chunk_id"] == "guide:1"
    page = client.post("/", data={"question": "example", "config": "hybrid"}).text
    assert "An example passage." in page and "guide.pdf#page=2" in page


@pytest.mark.parametrize(
    ("openai_key", "anthropic_key", "provider", "model"),
    [
        ("openai-test", None, "openai", "gpt-4o-mini"),
        (None, "anthropic-test", "anthropic", "claude-opus-5"),
        ("openai-test", "anthropic-test", "anthropic", "claude-opus-5"),
    ],
)
def test_answer_and_judge_route_to_available_provider(
    monkeypatch, result, openai_key, anthropic_key, provider, model
):
    monkeypatch.setattr(settings, "openai_api_key", openai_key)
    monkeypatch.setattr(settings, "anthropic_api_key", anthropic_key)
    config = load_config("dense_section")
    assert (config.generation.provider, config.generation.model) == (provider, model)
    retriever = Mock(config=config)
    retriever.retrieve.return_value = result
    monkeypatch.setattr(service, "get_retriever", lambda name: retriever)
    monkeypatch.setattr(service, "log_query", Mock())
    sdk = Mock()
    grounded = GroundedAnswer(answered=True, answer="Example [1].", citations=[1])
    expected = Verdict(accurate=True, faithful=True, reason="supported")
    if provider == "openai":
        monkeypatch.setattr("openai.OpenAI", sdk)
        parse = sdk.return_value.beta.chat.completions.parse
        parse.side_effect = [
            Mock(
                usage=Mock(prompt_tokens=5, completion_tokens=3),
                choices=[Mock(message=Mock(parsed=obj))],
            )
            for obj in (grounded, expected)
        ]
        schema_arg = "response_format"
    else:
        monkeypatch.setattr("anthropic.Anthropic", sdk)
        parse = sdk.return_value.messages.parse
        parse.side_effect = [
            Mock(usage=Mock(input_tokens=5, output_tokens=3), parsed_output=obj)
            for obj in (grounded, expected)
        ]
        schema_arg = "output_format"
    answer, retrieved = service.ask("example", config.name)
    assert answer.answered and answer.input_tokens == 5
    label = Label(query_id="q1", question="example", gold=[])
    assert judge_answer(label, answer, retrieved, config.generation) == expected
    assert [c.kwargs[schema_arg] for c in parse.call_args_list] == [GroundedAnswer, Verdict]
    assert all(c.kwargs["model"] == model for c in parse.call_args_list)
    assert all(c.kwargs["api_key"] == config.generation.api_key for c in sdk.call_args_list)


def test_explicit_provider_preserves_custom_model(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-test")
    monkeypatch.setattr(settings, "openai_api_key", "openai-test")
    selected = GenerationConfig(provider="openai", model="custom-model")
    assert (selected.provider, selected.model, selected.api_key) == (
        "openai",
        "custom-model",
        "openai-test",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "example", "config": "../configs/hybrid.yaml"},
        {"question": "   "},
        {"question": "example", "age_bands": ["invalid"]},
    ],
)
def test_api_rejects_invalid_input(payload):
    assert TestClient(app).post("/api/ask", json=payload).status_code == 422


def test_empty_retrieval_never_calls_provider(monkeypatch, result):
    result.results.clear()
    monkeypatch.setattr(
        "babyinfoquick.answer.generate.build_client", Mock(side_effect=AssertionError)
    )
    answer = generate("example", result, load_config("hybrid"), time.perf_counter())
    assert answer.text == NO_ANSWER_TEXT and not answer.answered


@pytest.mark.parametrize(
    ("text", "markers", "valid"),
    [
        ("Example [1].", [1], True),
        ("Example.", [], False),
        ("Example [2].", [2], False),
        ("Example [1] and [2].", [1], False),
        ("[1]", [1], False),
    ],
)
def test_only_resolvable_inline_citations_are_served(result, text, markers, valid):
    parsed = GroundedAnswer(answered=True, answer=text, citations=markers)
    answer = _to_answer(parsed, Usage(), result, load_config("hybrid"), time.perf_counter())
    assert answer.answered is valid
    assert [c.chunk_id for c in answer.citations] == (["guide:1"] if valid else [])


def test_judged_run_separates_refusal_controls_from_answerable_queries(
    monkeypatch, result, tmp_path
):
    retriever = Mock()
    retriever.retrieve.return_value = result
    label = Label(query_id="q1", question="example", gold=[GoldPassage(doc_id="guide", page=2)])
    control = Label(query_id="r1", question="outside", gold=[])
    monkeypatch.setattr(runner, "Retriever", lambda config: retriever)
    monkeypatch.setattr(runner, "read_chunks", lambda strategy: [result.results[0].chunk])
    monkeypatch.setattr(
        runner, "load_labels", lambda path=None: [label] if path is None else [control]
    )
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        runner,
        "judge_answer",
        Mock(return_value=Verdict(accurate=True, faithful=True, reason="supported")),
    )

    def generate_answer(question, trimmed, config, started):
        parsed = GroundedAnswer(answered=question == "example", answer="Sample [1].", citations=[1])
        return _to_answer(parsed, Usage(input_tokens=10), trimmed, config, started)

    monkeypatch.setattr(runner, "generate", generate_answer)
    report = runner.run_config("hybrid", judge=True)
    assert report.retrieval.n == report.answer.n == 1
    assert report.answer.citation_precision == report.answer.answer_accuracy == 1
    assert report.answer.refusal_control_n == report.answer.refusal_control_rate == 1
    assert report.operational.mean_input_tokens == 10
