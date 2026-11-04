from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from babyinfoquick.age import AgeBand


class SourceDoc(BaseModel):
    doc_id: str
    title: str
    url: str
    publisher: str
    doc_type: str
    age_bands: list[AgeBand] = Field(default_factory=list)
    retrieved_at: date | None = None
    local_path: str | None = None
    sha256: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    section: str | None = None
    page_start: int
    page_end: int
    token_estimate: int
    title: str
    url: str
    publisher: str
    doc_type: str
    age_bands: list[AgeBand] = Field(default_factory=list)
    retrieved_at: date | None = None


class Scored(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    marker: int
    chunk_id: str
    title: str
    publisher: str
    url: str
    page_start: int
    page_end: int


class Answer(BaseModel):
    answered: bool
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_passages: list[Scored] = Field(default_factory=list)
    age_bands: list[AgeBand] = Field(default_factory=list)
    disclaimer: str
    config_name: str
    retrieval_ms: float
    total_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
