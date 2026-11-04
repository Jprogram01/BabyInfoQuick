from __future__ import annotations

from pydantic import BaseModel, Field

from babyinfoquick.answer.generate import format_passages
from babyinfoquick.answer.llm import build_client
from babyinfoquick.config import GenerationConfig
from babyinfoquick.evaluation.labels import Label
from babyinfoquick.retrieval.pipeline import RetrievalResult
from babyinfoquick.types import Answer

JUDGE_SYSTEM = """You grade answers from a retrieval system that answers parenting questions from official pediatric guidance.

You see the question, the passages the system retrieved, and the answer it produced.

accurate: true if the answer correctly answers the question given the passages. An honest refusal counts as accurate when the passages genuinely do not contain the answer.
faithful: true if every factual claim in the answer is supported by the passages. An answer containing anything not in the passages is unfaithful, even if it is true in general.

Judge only against the passages. Do not use your own medical knowledge."""


class Verdict(BaseModel):
    accurate: bool = Field(description="Does the answer correctly address the question?")
    faithful: bool = Field(description="Does every claim trace to a retrieved passage?")
    reason: str = Field(description="One sentence explaining the verdict.")


def judge_answer(
    label: Label, answer: Answer, result: RetrievalResult, config: GenerationConfig
) -> Verdict:
    client = build_client(config.model_copy(update={"max_tokens": 500}))
    user = (
        f"Question: {label.question}\n\n"
        f"Passages:\n\n{format_passages(result)}\n\n"
        f"Answer produced by the system:\n{answer.text}"
    )
    verdict, _ = client.complete(JUDGE_SYSTEM, user, Verdict)
    if verdict is None:
        raise RuntimeError(f"The judge returned no verdict for {label.query_id}.")
    return verdict
