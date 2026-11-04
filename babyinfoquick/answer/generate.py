from __future__ import annotations

import re
import time

from babyinfoquick.answer.llm import GroundedAnswer, Usage, build_client
from babyinfoquick.config import PipelineConfig
from babyinfoquick.retrieval.pipeline import RetrievalResult
from babyinfoquick.types import Answer, Citation

DISCLAIMER = (
    "Informational only, not medical advice. Contact your pediatrician for anything "
    "specific to your child or anything urgent."
)

SYSTEM_PROMPT = """You answer questions from parents about infant and toddler care, using only the numbered passages provided.

Rules:
- Use only the passages. Never use outside knowledge, even if you are confident.
- If the passages do not answer the question, set answered to false and leave the answer empty.
- Cite every claim with an inline marker matching the passage number, like [1] or [2].
- Match the age band the parent asked about. If the passages cover a different age, say so rather than generalizing.
- Never give a diagnosis, a medication dose, or emergency instructions. If the question calls for one, set answered to false.
- Write plainly, for a tired parent. Short sentences.
- Do not add a disclaimer; one is appended automatically."""

NO_ANSWER_TEXT = (
    "I could not find guidance on that in the source documents. That may mean the "
    "question is outside what these public documents cover, or that it needs a "
    "clinician. Please ask your pediatrician."
)


def format_passages(result: RetrievalResult) -> str:
    blocks = []
    for index, scored in enumerate(result.results, start=1):
        chunk = scored.chunk
        header = f"[{index}] {chunk.title} ({chunk.publisher}, p. {chunk.page_start})"
        if chunk.section:
            header += f" - section: {chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, result: RetrievalResult) -> str:
    age_note = ""
    if result.age_bands:
        bands = ", ".join(b.value for b in result.age_bands)
        age_note = f"\nAge band requested: {bands}"
    return f"Question: {question}{age_note}\n\nPassages:\n\n{format_passages(result)}"


def generate(
    question: str, result: RetrievalResult, config: PipelineConfig, started: float
) -> Answer:
    if not result.results:
        return Answer(
            answered=False,
            text=NO_ANSWER_TEXT,
            citations=[],
            age_bands=result.age_bands,
            disclaimer=DISCLAIMER,
            config_name=config.name,
            retrieval_ms=result.retrieval_ms,
            total_ms=(time.perf_counter() - started) * 1000,
        )

    client = build_client(config.generation)
    parsed, usage = client.complete(
        SYSTEM_PROMPT, build_user_prompt(question, result), GroundedAnswer
    )

    if parsed is None:
        parsed = GroundedAnswer(answered=False, answer="")
    return _to_answer(parsed, usage, result, config, started)


def _to_answer(
    parsed: GroundedAnswer,
    usage: Usage,
    result: RetrievalResult,
    config: PipelineConfig,
    started: float,
) -> Answer:
    citations: list[Citation] = []
    inline = [int(marker) for marker in re.findall(r"\[(\d+)\]", parsed.answer)]
    answered = (
        parsed.answered
        and bool(inline)
        and set(inline) == set(parsed.citations)
        and all(1 <= marker <= len(result.results) for marker in inline)
        and bool(re.sub(r"\[\d+\]", "", parsed.answer).strip())
    )
    if answered:
        for marker in dict.fromkeys(inline):
            chunk = result.results[marker - 1].chunk
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    publisher=chunk.publisher,
                    url=chunk.url,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )

    return Answer(
        answered=answered,
        text=parsed.answer if answered else NO_ANSWER_TEXT,
        citations=citations,
        age_bands=result.age_bands,
        disclaimer=DISCLAIMER,
        config_name=config.name,
        retrieval_ms=result.retrieval_ms,
        total_ms=(time.perf_counter() - started) * 1000,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
