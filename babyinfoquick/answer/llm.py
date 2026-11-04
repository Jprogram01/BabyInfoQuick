from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel, Field

from babyinfoquick.config import GenerationConfig

PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


class GroundedAnswer(BaseModel):
    answered: bool = Field(
        description="True only if the passages contain enough information to answer."
    )
    answer: str = Field(
        description=(
            "The answer in plain language for a parent, with inline citation markers "
            "like [1] or [2] referring to the numbered passages. Empty if answered is false."
        )
    )
    citations: list[int] = Field(
        default_factory=list,
        description="Passage numbers actually cited in the answer.",
    )


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, model: str) -> float:
        rates = PRICING_PER_MTOK.get(model)
        if rates is None:
            return 0.0
        return (self.input_tokens * rates[0] + self.output_tokens * rates[1]) / 1_000_000


Output = TypeVar("Output", bound=BaseModel)


class LLMClient(Protocol):
    def complete(
        self, system: str, user: str, schema: type[Output]
    ) -> tuple[Output | None, Usage]: ...


class AnthropicClient:
    def __init__(self, config: GenerationConfig):
        import anthropic

        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)

    def complete(self, system: str, user: str, schema: type[Output]) -> tuple[Output | None, Usage]:
        response = self.client.messages.parse(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            output_format=schema,
        )
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.parsed_output, usage


class OpenAIClient:
    def __init__(self, config: GenerationConfig):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(api_key=config.api_key)

    def complete(self, system: str, user: str, schema: type[Output]) -> tuple[Output | None, Usage]:
        response = self.client.beta.chat.completions.parse(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        return response.choices[0].message.parsed, usage


def build_client(config: GenerationConfig) -> LLMClient:
    if config.provider == "openai":
        return OpenAIClient(config)
    return AnthropicClient(config)
