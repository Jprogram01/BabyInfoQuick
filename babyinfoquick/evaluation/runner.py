from __future__ import annotations

import json
import time

from babyinfoquick.answer.generate import generate
from babyinfoquick.answer.llm import PRICING_PER_MTOK, Usage
from babyinfoquick.config import EVAL_DIR, list_configs, load_config
from babyinfoquick.corpus.chunk import read_chunks
from babyinfoquick.evaluation.judge import judge_answer
from babyinfoquick.evaluation.labels import count_relevant_chunks, is_relevant, load_labels
from babyinfoquick.evaluation.metrics import (
    AnswerMetrics,
    ConfigReport,
    OperationalMetrics,
    RetrievalMetrics,
    percentile,
)
from babyinfoquick.retrieval.pipeline import Retriever

RESULTS_DIR = EVAL_DIR / "results"
EVAL_TOP_K = 10


def run_config(config_name: str, judge: bool = False) -> ConfigReport:
    config = load_config(config_name)
    eval_config = config.model_copy(deep=True)
    eval_config.retrieval.final_top_k = max(EVAL_TOP_K, config.retrieval.final_top_k)
    retriever = Retriever(eval_config)

    labels = load_labels()
    relevant_counts = count_relevant_chunks(labels, read_chunks(config.chunking.strategy))
    print(f"\n=== {config.name} === ({len(labels)} labeled queries)")

    runs: list[list[bool]] = []
    runs_age_sensitive: list[list[bool]] = []
    runs_age_agnostic: list[list[bool]] = []
    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    usages: list[Usage] = []

    answered = 0
    citation_hits = 0
    citation_total = 0
    age_correct = 0
    age_queries = 0
    traces = []
    judged: list[dict] = []

    for label, total_relevant in zip(labels, relevant_counts, strict=True):
        started = time.perf_counter()
        result = retriever.retrieve(label.question, label.age_bands or None)
        scored, elapsed = result.results, result.retrieval_ms
        relevance = [is_relevant(s.chunk, label.gold) for s in scored]
        traces.append(
            {
                "query_id": label.query_id,
                "chunk_ids": [s.chunk.chunk_id for s in scored],
                "relevance": relevance,
                "total_relevant": total_relevant,
            }
        )
        runs.append(relevance)
        retrieval_ms.append(elapsed)
        (runs_age_sensitive if label.age_sensitive else runs_age_agnostic).append(relevance)

        if label.age_bands:
            age_queries += 1
            wanted = set(label.age_bands)
            top = scored[: config.retrieval.final_top_k]
            if top and all(wanted.intersection(s.chunk.age_bands) for s in top):
                age_correct += 1

        if judge:
            trimmed = result.model_copy(update={"results": scored[: config.retrieval.final_top_k]})
            answer = generate(label.question, trimmed, config, started)
            total_ms.append(answer.total_ms)
            usages.append(
                Usage(input_tokens=answer.input_tokens, output_tokens=answer.output_tokens)
            )
            if answer.answered:
                answered += 1
                for citation in answer.citations:
                    citation_total += 1
                    index = citation.marker - 1
                    if 0 <= index < len(relevance) and relevance[index]:
                        citation_hits += 1
            verdict = judge_answer(label, answer, trimmed, config.generation)
            judged.append(
                {
                    "query_id": label.query_id,
                    "question": label.question,
                    "answered": answer.answered,
                    "accurate": verdict.accurate,
                    "faithful": verdict.faithful,
                    "reason": verdict.reason,
                }
            )

    n = len(labels)
    refusals = []
    if judge:
        for control in load_labels(EVAL_DIR / "refusal_controls.jsonl"):
            started = time.perf_counter()
            result = retriever.retrieve(control.question)
            trimmed = result.model_copy(
                update={"results": result.results[: config.retrieval.final_top_k]}
            )
            response = generate(control.question, trimmed, config, started)
            refusals.append({"query_id": control.query_id, "refused": not response.answered})
    answer_metrics = AnswerMetrics(
        n=n,
        answer_rate=answered / n if judge and n else None,
        citation_precision=citation_hits / citation_total if citation_total else None,
        age_band_correctness=age_correct / age_queries if age_queries else None,
        faithfulness=(sum(j["faithful"] for j in judged) / len(judged)) if judged else None,
        answer_accuracy=(sum(j["accurate"] for j in judged) / len(judged)) if judged else None,
        refusal_control_n=len(refusals),
        refusal_control_rate=sum(row["refused"] for row in refusals) / len(refusals)
        if refusals
        else None,
    )

    mean_in = sum(u.input_tokens for u in usages) / len(usages) if usages else 0.0
    mean_out = sum(u.output_tokens for u in usages) / len(usages) if usages else 0.0
    rates = PRICING_PER_MTOK.get(config.generation.model)
    operational = OperationalMetrics(
        p50_retrieval_ms=percentile(retrieval_ms, 50),
        p95_retrieval_ms=percentile(retrieval_ms, 95),
        p50_total_ms=percentile(total_ms, 50) if judge else None,
        p95_total_ms=percentile(total_ms, 95) if judge else None,
        mean_input_tokens=mean_in if judge else None,
        mean_output_tokens=mean_out if judge else None,
        mean_cost_usd=(mean_in * rates[0] + mean_out * rates[1]) / 1_000_000
        if judge and rates
        else None,
    )

    report = ConfigReport(
        config_name=config.name,
        description=config.description,
        retrieval=RetrievalMetrics.from_runs(runs, relevant_counts),
        retrieval_age_sensitive=RetrievalMetrics.from_runs(
            runs_age_sensitive,
            [n for label, n in zip(labels, relevant_counts) if label.age_sensitive],
        ),
        retrieval_age_agnostic=RetrievalMetrics.from_runs(
            runs_age_agnostic,
            [n for label, n in zip(labels, relevant_counts) if not label.age_sensitive],
        ),
        answer=answer_metrics,
        operational=operational,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{config.name}_retrieval.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in traces), encoding="utf-8"
    )
    (RESULTS_DIR / f"{config.name}.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    if judged:
        with open(RESULTS_DIR / f"{config.name}_judged.jsonl", "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in judged)
        (RESULTS_DIR / f"{config.name}_refusals.json").write_text(
            json.dumps(refusals, indent=2), encoding="utf-8"
        )

    _print_report(report)
    return report


def _print_report(report: ConfigReport) -> None:
    r = report.retrieval
    print(
        f"  recall@1  {r.recall_at_1:.3f}   recall@5 {r.recall_at_5:.3f}   recall@10 {r.recall_at_10:.3f}"
    )
    print(f"  MRR       {r.mrr:.3f}   nDCG@10  {r.ndcg_at_10:.3f}")
    print(
        f"  age-sensitive recall@5 {report.retrieval_age_sensitive.recall_at_5:.3f}"
        f" (n={report.retrieval_age_sensitive.n})   "
        f"age-agnostic {report.retrieval_age_agnostic.recall_at_5:.3f}"
        f" (n={report.retrieval_age_agnostic.n})"
    )
    print(f"  age-band correctness {report.answer.age_band_correctness}")
    if report.answer.answer_accuracy is not None:
        print(
            f"  answer accuracy {report.answer.answer_accuracy:.3f}   "
            f"faithfulness {report.answer.faithfulness:.3f}   "
            f"citation precision {report.answer.citation_precision}"
        )
    o = report.operational
    print(f"  p95 retrieval {o.p95_retrieval_ms:.0f} ms")
    if o.p95_total_ms is not None:
        print(f"  p95 total {o.p95_total_ms:.0f} ms")
    if report.answer.refusal_control_n:
        print(
            f"  refusal controls {report.answer.refusal_control_rate:.1%} "
            f"(n={report.answer.refusal_control_n})"
        )


ABLATION_ORDER = ["dense_fixed", "dense_section", "hybrid", "hybrid_rerank"]


def run_all(judge: bool = False) -> list[ConfigReport]:
    names = [n for n in ABLATION_ORDER if n in list_configs()]
    names += [n for n in list_configs() if n not in names]
    reports = [run_config(name, judge=judge) for name in names]
    print("\n" + render_table(reports))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ablation.md").write_text(render_table(reports), encoding="utf-8")
    (RESULTS_DIR / "reports.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in reports], indent=2), encoding="utf-8"
    )
    return reports


def render_table(reports: list[ConfigReport]) -> str:
    header = (
        "| Config | Recall@5 | MRR | Age-sensitive R@5 | Age-agnostic R@5 | "
        "Answer accuracy | p95 retrieval | Cost/query |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = []
    for report in reports:
        accuracy = (
            f"{report.answer.answer_accuracy:.3f}"
            if report.answer.answer_accuracy is not None
            else "not run"
        )
        cost = report.operational.mean_cost_usd
        cost_text = f"${cost:.5f}" if cost is not None else "not measured"
        rows.append(
            f"| {report.config_name} | {report.retrieval.recall_at_5:.3f} | "
            f"{report.retrieval.mrr:.3f} | {report.retrieval_age_sensitive.recall_at_5:.3f} | "
            f"{report.retrieval_age_agnostic.recall_at_5:.3f} | {accuracy} | "
            f"{report.operational.p95_retrieval_ms:.0f} ms | {cost_text} |"
        )
    return "\n".join([header, *rows])
