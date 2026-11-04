from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from babyinfoquick.config import PARSED_DIR, ensure_dirs
from babyinfoquick.types import SourceDoc

MIN_CHARS_PER_PAGE = 120


class Page(BaseModel):
    page: int
    text: str
    ocr: bool = False


class ParsedDoc(BaseModel):
    doc_id: str
    pages: list[Page]

    @property
    def n_ocr(self) -> int:
        return sum(1 for p in self.pages if p.ocr)


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")
_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = _HYPHEN_RE.sub(r"\1\2", text)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def _extract_pymupdf(path: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(path) as doc:
        return [page.get_text("text") for page in doc]


def _ocr_page(path: Path, page_index: int) -> str:
    import pymupdf
    import pytesseract
    from PIL import Image

    with pymupdf.open(path) as doc:
        page = doc[page_index]
        pixmap = page.get_pixmap(dpi=200)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return pytesseract.image_to_string(image)


def parse_pdf(path: Path, allow_ocr: bool = True) -> list[Page]:
    raw_pages = _extract_pymupdf(path)
    pages: list[Page] = []
    for index, raw in enumerate(raw_pages):
        text = normalize(raw)
        used_ocr = False
        if len(text) < MIN_CHARS_PER_PAGE and allow_ocr:
            try:
                ocr_text = normalize(_ocr_page(path, index))
                if len(ocr_text) > len(text):
                    text = ocr_text
                    used_ocr = True
            except (OSError, RuntimeError) as exc:
                print(f"  ocr failed on page {index + 1}: {exc}")
        if text:
            pages.append(Page(page=index + 1, text=text, ocr=used_ocr))
    return pages


def parse_all(docs: list[SourceDoc], allow_ocr: bool = True) -> list[ParsedDoc]:
    ensure_dirs()
    parsed: list[ParsedDoc] = []
    for doc in docs:
        if not doc.local_path:
            continue
        out_path = PARSED_DIR / f"{doc.doc_id}.json"
        pages = parse_pdf(Path(doc.local_path), allow_ocr=allow_ocr)
        parsed_doc = ParsedDoc(doc_id=doc.doc_id, pages=pages)
        out_path.write_text(parsed_doc.model_dump_json(indent=2), encoding="utf-8")
        chars = sum(len(p.text) for p in pages)
        print(f"parsed  {doc.doc_id}  {len(pages)} pages  {chars} chars  ocr={parsed_doc.n_ocr}")
        parsed.append(parsed_doc)
    return parsed


def load_parsed(doc_id: str) -> ParsedDoc:
    path = PARSED_DIR / f"{doc_id}.json"
    return ParsedDoc.model_validate(json.loads(path.read_text(encoding="utf-8")))
