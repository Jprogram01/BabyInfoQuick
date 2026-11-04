from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
BM25_DIR = DATA_DIR / "bm25"
LOGS_DIR = DATA_DIR / "logs"
CONFIGS_DIR = REPO_ROOT / "configs"
EVAL_DIR = REPO_ROOT / "eval"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    default_config: str = "dense_section"
    query_logging: bool = True
    max_asks_per_minute: int = Field(default=0, ge=0)


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "section"] = "section"
    target_tokens: int = 350
    overlap_tokens: int = 60
    max_tokens: int = 600
    min_tokens: int = 40


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-base-en-v1.5"
    revision: str | None = None
    dim: int = 768
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32
    normalize: bool = True


class RetrievalConfig(BaseModel):
    dense: bool = True
    lexical: bool = False
    dense_top_k: int = 30
    lexical_top_k: int = 30
    fusion: Literal["rrf", "none"] = "rrf"
    rrf_k: int = 60
    age_filter: Literal["pre", "post", "off"] = "pre"
    final_top_k: int = 5
    min_score: float = 0.0


class RerankConfig(BaseModel):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-base"
    revision: str | None = None
    top_k: int = 20
    batch_size: int = 16
    max_length: int = 512


class GenerationConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-opus-5"
    max_tokens: int = 900
    temperature: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def select_provider(cls, values):
        if not isinstance(values, dict):
            return values
        values = values.copy()
        if (
            "provider" not in values and "model" not in values
            and settings.openai_api_key and not settings.anthropic_api_key
        ):
            values["provider"] = "openai"
        if values.get("provider") == "openai":
            values.setdefault("model", "gpt-4o-mini")
        return values

    @property
    def api_key(self) -> str | None:
        return settings.openai_api_key if self.provider == "openai" else settings.anthropic_api_key


class PipelineConfig(BaseModel):
    name: str
    description: str = ""
    collection: str = "babyinfoquick"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


def load_config(name_or_path: str) -> PipelineConfig:
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIGS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"No config {name_or_path!r}. Available: {available}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.setdefault("name", path.stem)
    return PipelineConfig.model_validate(data)


def list_configs() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))


def ensure_dirs() -> None:
    for directory in (RAW_DIR, PARSED_DIR, CHUNKS_DIR, BM25_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
