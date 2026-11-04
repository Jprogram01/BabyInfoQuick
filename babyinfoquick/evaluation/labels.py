from __future__ import annotations

import json
import re
from collections import defaultdict

from pydantic import BaseModel, Field

from babyinfoquick.age import AgeBand
from babyinfoquick.config import EVAL_DIR
from babyinfoquick.types import Chunk

LABELS_PATH = EVAL_DIR / "labels.jsonl"

_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return " ".join(_NORM_RE.sub(" ", text.lower()).split())


class GoldPassage(BaseModel):
    doc_id: str
    snippet: str = ""
    page: int | None = None

    def model_post_init(self, _context) -> None:
        if not self.snippet and self.page is None:
            raise ValueError(f"gold passage for {self.doc_id} needs a snippet or a page")


class Label(BaseModel):
    query_id: str
    question: str
    age_bands: list[AgeBand] = Field(default_factory=list)
    age_sensitive: bool = False
    gold: list[GoldPassage]
    notes: str | None = None


def load_labels(path=LABELS_PATH) -> list[Label]:
    if not path.exists():
        raise FileNotFoundError(
            f"No labels at {path}. Weekend 2 is hand-labeling these; see eval/README.md"
        )
    labels: list[Label] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            labels.append(Label.model_validate(json.loads(line)))
    if not labels or len({label.query_id for label in labels}) != len(labels):
        raise ValueError("The label set must be nonempty with unique query IDs.")
    return labels


def is_relevant(chunk: Chunk, gold: list[GoldPassage]) -> bool:
    for passage in gold:
        if passage.doc_id != chunk.doc_id:
            continue
        if passage.snippet and normalize(passage.snippet) in normalize(chunk.text):
            return True
        if not passage.snippet and passage.page in range(chunk.page_start, chunk.page_end + 1):
            return True
    return False


def count_relevant_chunks(labels: list[Label], chunks: list[Chunk]) -> list[int]:
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)
    counts = []
    for label in labels:
        doc_ids = {gold.doc_id for gold in label.gold}
        count = sum(
            is_relevant(chunk, label.gold) for doc_id in doc_ids for chunk in by_doc[doc_id]
        )
        counts.append(count)
    return counts


def validate_passages(labels: list[Label]) -> None:
    from babyinfoquick.corpus.parse import load_parsed

    documents = {}
    for label in labels:
        for gold in label.gold:
            if gold.doc_id not in documents:
                documents[gold.doc_id] = {
                    p.page: normalize(p.text) for p in load_parsed(gold.doc_id).pages
                }
            pages = documents[gold.doc_id]
            if gold.page is not None and gold.page not in pages:
                raise ValueError(f"{label.query_id}: missing {gold.doc_id} page {gold.page}")
            candidates = [pages[gold.page]] if gold.page is not None else pages.values()
            if gold.snippet and not any(normalize(gold.snippet) in page for page in candidates):
                raise ValueError(
                    f"{label.query_id}: snippet missing from {gold.doc_id} p.{gold.page}"
                )
