from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi

from babyinfoquick.age import AgeBand
from babyinfoquick.config import BM25_DIR, ensure_dirs
from babyinfoquick.types import Chunk, Scored

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    def __init__(self, chunks: list[Chunk], bm25: BM25Okapi):
        self.chunks = chunks
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks: list[Chunk]) -> BM25Store:
        corpus = [tokenize(f"{c.section or ''} {c.text}") for c in chunks]
        return cls(chunks, BM25Okapi(corpus))

    def save(self, name: str) -> None:
        ensure_dirs()
        path = BM25_DIR / f"{name}.pkl"
        with open(path, "wb") as handle:
            pickle.dump({"chunks": [c.model_dump(mode="json") for c in self.chunks],
                         "bm25": self.bm25}, handle)

    @classmethod
    def load(cls, name: str) -> BM25Store:
        path = BM25_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index at {path}. Run: biq index")
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        chunks = [Chunk.model_validate(c) for c in data["chunks"]]
        return cls(chunks, data["bm25"])

    def search(
        self, query: str, top_k: int, age_bands: list[AgeBand] | None = None
    ) -> list[Scored]:
        scores = self.bm25.get_scores(tokenize(query))
        allowed = set(age_bands) if age_bands else None

        ranked = []
        for index, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = self.chunks[index]
            if allowed is not None and not allowed.intersection(chunk.age_bands):
                continue
            ranked.append((score, chunk))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Scored(chunk=chunk, score=float(score), lexical_score=float(score))
            for score, chunk in ranked[:top_k]
        ]
