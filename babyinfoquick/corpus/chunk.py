from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from babyinfoquick.age import AgeBand, extract_age_bands
from babyinfoquick.config import CHUNKS_DIR, ChunkingConfig, ensure_dirs
from babyinfoquick.corpus.parse import ParsedDoc
from babyinfoquick.types import Chunk, SourceDoc

_SENT_RE = re.compile(r"(?<=[.!?:;])\s+|\n+")
_BULLET_RE = re.compile(r"^\s*([•▪●◦‣▸○–—\-\*■·]|\d+[.)])\s+")
_NUMBERED_HEAD_RE = re.compile(r"^\s*(chapter|section|part)\s+\d+", re.IGNORECASE)
_MID_SENTENCE_RE = re.compile(r"[.!?]\s+\S")
_EDGE_NUM_RE = re.compile(r"^\s*[\divxlc]+\s+|\s+[\divxlc]+\s*$", re.IGNORECASE)
RUNNING_HEADER_MIN_PAGES = 3
_AGE_HEAD_RE = re.compile(
    r"^\s*(your (baby|child|toddler|infant) at\b|milestones? (at|by)\b|\d+\s*(months?|years?)\b)",
    re.IGNORECASE,
)

MAX_HEADING_CHARS = 90


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if _BULLET_RE.match(stripped):
        return False
    if _NUMBERED_HEAD_RE.match(stripped) or _AGE_HEAD_RE.match(stripped):
        return True
    if stripped.endswith((".", ",", ";")):
        return False
    words = stripped.split()
    if len(words) > 5 and _MID_SENTENCE_RE.search(stripped):
        return False
    if len(words) > 12:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    if sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        return True
    if stripped.endswith("?") and len(words) <= 12:
        return True
    capitalized = sum(1 for w in words if w[:1].isupper())
    return capitalized / len(words) >= 0.6 and len(words) >= 2


def infer_age_bands(
    section: str | None, text: str, doc_bands: list[AgeBand]
) -> list[AgeBand]:
    if section:
        from_section = extract_age_bands(section)
        if from_section:
            return from_section
    if len(doc_bands) <= 4:
        return doc_bands
    from_text = extract_age_bands(text[:400])
    return from_text or doc_bands


class Block(BaseModel):
    section: str | None
    text: str
    page_start: int
    page_end: int


def _header_key(line: str) -> str:
    return _EDGE_NUM_RE.sub("", line.strip()).strip().lower()


def find_running_headers(parsed: ParsedDoc) -> set[str]:
    pages_seen: dict[str, set[int]] = {}
    for page in parsed.pages:
        for line in page.text.split("\n"):
            stripped = line.strip()
            if not stripped or len(stripped) > MAX_HEADING_CHARS:
                continue
            if _AGE_HEAD_RE.match(stripped):
                continue
            key = _header_key(stripped)
            if len(key) < 6:
                continue
            pages_seen.setdefault(key, set()).add(page.page)
    return {k for k, pages in pages_seen.items() if len(pages) >= RUNNING_HEADER_MIN_PAGES}


def is_running_header(line: str, headers: set[str]) -> bool:
    stripped = line.strip()
    if _AGE_HEAD_RE.match(stripped):
        return False
    return _header_key(stripped) in headers


def split_into_blocks(parsed: ParsedDoc) -> list[Block]:
    headers = find_running_headers(parsed)
    blocks: list[Block] = []
    section: str | None = None
    buffer: list[str] = []
    page_start = parsed.pages[0].page if parsed.pages else 1
    page_end = page_start

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            blocks.append(
                Block(section=section, text=body, page_start=page_start, page_end=page_end)
            )
        buffer = []

    for page in parsed.pages:
        for line in page.text.split("\n"):
            if is_running_header(line, headers):
                continue
            if is_heading(line):
                flush()
                section = line.strip()
                page_start = page.page
            elif line.strip():
                if not buffer:
                    page_start = page.page
                buffer.append(line.strip())
            page_end = page.page
    flush()
    return blocks


def _pack(units: list[str], config: ChunkingConfig) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > config.target_tokens:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(prev)
                if overlap_tokens + prev_tokens > config.overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_tokens += prev_tokens
            current = overlap
            current_tokens = overlap_tokens
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_section_aware(
    parsed: ParsedDoc, doc: SourceDoc, config: ChunkingConfig
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block_index, block in enumerate(split_into_blocks(parsed)):
        units = [u.strip() for u in _SENT_RE.split(block.text) if u.strip()]
        for part_index, body in enumerate(_pack(units, config)):
            if estimate_tokens(body) < config.min_tokens:
                continue
            chunks.append(
                _build_chunk(
                    doc=doc,
                    text=body,
                    section=block.section,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    suffix=f"s{block_index:04d}p{part_index:02d}",
                )
            )
    return chunks


def chunk_fixed(parsed: ParsedDoc, doc: SourceDoc, config: ChunkingConfig) -> list[Chunk]:
    words: list[tuple[str, int]] = []
    for page in parsed.pages:
        for word in page.text.split():
            words.append((word, page.page))

    stride = max(1, int((config.target_tokens - config.overlap_tokens) / 1.3))
    window = max(1, int(config.target_tokens / 1.3))

    chunks: list[Chunk] = []
    index = 0
    part = 0
    while index < len(words):
        window_words = words[index : index + window]
        if not window_words:
            break
        body = " ".join(w for w, _ in window_words)
        if estimate_tokens(body) >= config.min_tokens:
            chunks.append(
                _build_chunk(
                    doc=doc,
                    text=body,
                    section=None,
                    page_start=window_words[0][1],
                    page_end=window_words[-1][1],
                    suffix=f"f{part:05d}",
                )
            )
        part += 1
        index += stride
    return chunks


def _build_chunk(
    doc: SourceDoc,
    text: str,
    section: str | None,
    page_start: int,
    page_end: int,
    suffix: str,
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}:{suffix}",
        doc_id=doc.doc_id,
        text=text,
        section=section,
        page_start=page_start,
        page_end=page_end,
        token_estimate=estimate_tokens(text),
        title=doc.title,
        url=doc.url,
        publisher=doc.publisher,
        doc_type=doc.doc_type,
        age_bands=infer_age_bands(section, text, doc.age_bands),
        retrieved_at=doc.retrieved_at,
    )


def chunk_doc(parsed: ParsedDoc, doc: SourceDoc, config: ChunkingConfig) -> list[Chunk]:
    if config.strategy == "fixed":
        return chunk_fixed(parsed, doc, config)
    return chunk_section_aware(parsed, doc, config)


def chunks_path(strategy: str) -> Path:
    return CHUNKS_DIR / f"chunks_{strategy}.jsonl"


def write_chunks(chunks: list[Chunk], strategy: str) -> None:
    ensure_dirs()
    path = chunks_path(strategy)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(chunk.model_dump_json() + "\n" for chunk in chunks)


def read_chunks(strategy: str) -> list[Chunk]:
    path = chunks_path(strategy)
    if not path.exists():
        raise FileNotFoundError(f"No chunks for strategy {strategy!r}. Run: biq index")
    with open(path, encoding="utf-8") as handle:
        return [Chunk.model_validate(json.loads(line)) for line in handle if line.strip()]
