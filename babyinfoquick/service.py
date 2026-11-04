from __future__ import annotations

import time
from functools import lru_cache

from babyinfoquick.age import AgeBand
from babyinfoquick.answer.generate import DISCLAIMER, generate
from babyinfoquick.config import PipelineConfig, load_config
from babyinfoquick.querylog import log_query
from babyinfoquick.retrieval.pipeline import RetrievalResult, Retriever
from babyinfoquick.types import Answer


@lru_cache(maxsize=8)
def get_retriever(config_name: str) -> Retriever:
    return Retriever(load_config(config_name))


def ask(
    question: str, config_name: str, age_bands: list[AgeBand] | None = None
) -> tuple[Answer, RetrievalResult]:
    started = time.perf_counter()
    retriever = get_retriever(config_name)
    result = retriever.retrieve(question, age_bands)
    generation = retriever.config.generation
    if result.results and not generation.api_key:
        keys = "OPENAI_API_KEY or ANTHROPIC_API_KEY"
        if "provider" in generation.model_fields_set or "model" in generation.model_fields_set:
            keys = f"{generation.provider.upper()}_API_KEY"
        answer = Answer(
            answered=False,
            text=f"Showing retrieved passages. Set {keys} to generate a cited answer.",
            citations=[],
            retrieved_passages=result.results,
            age_bands=result.age_bands,
            disclaimer=DISCLAIMER,
            config_name=retriever.config.name,
            retrieval_ms=result.retrieval_ms,
            total_ms=(time.perf_counter() - started) * 1000,
        )
    else:
        answer = generate(question, result, retriever.config, started)
    log_query(question, result, answer)
    return answer, result


def config_for(config_name: str) -> PipelineConfig:
    return get_retriever(config_name).config
