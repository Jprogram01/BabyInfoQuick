from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from babyinfoquick.age import bands_for_months
from babyinfoquick.config import RAW_DIR, ensure_dirs
from babyinfoquick.types import SourceDoc

SOURCES_PATH = Path(__file__).with_name("sources.yaml")
MANIFEST_PATH = RAW_DIR / "manifest.json"

USER_AGENT = "BabyInfoQuick/0.1 (personal learning project; contact via repo)"


def load_sources(path: Path = SOURCES_PATH) -> list[SourceDoc]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    docs = []
    for entry in data["sources"]:
        lo, hi = entry["age_months"]
        docs.append(
            SourceDoc(
                doc_id=entry["doc_id"],
                title=entry["title"],
                url=entry["url"],
                publisher=entry["publisher"],
                doc_type=entry["doc_type"],
                age_bands=bands_for_months(float(lo), float(hi)),
            )
        )
    return docs


def fetch_all(force: bool = False, timeout: float = 60.0) -> tuple[list[SourceDoc], list[dict]]:
    ensure_dirs()
    docs = load_sources()
    fetched: list[SourceDoc] = []
    failures: list[dict] = []

    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"}
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        for doc in docs:
            target = RAW_DIR / f"{doc.doc_id}.pdf"
            if target.exists() and not force:
                doc.local_path = str(target)
                doc.sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
                doc.retrieved_at = datetime.fromtimestamp(target.stat().st_mtime, UTC).date()
                fetched.append(doc)
                print(f"cached  {doc.doc_id}")
                continue

            try:
                response = client.get(doc.url)
                response.raise_for_status()
                content = response.content
                if not content.startswith(b"%PDF"):
                    raise ValueError(
                        f"not a PDF (content-type={response.headers.get('content-type')})"
                    )
                target.write_bytes(content)
                doc.local_path = str(target)
                doc.sha256 = hashlib.sha256(content).hexdigest()
                doc.retrieved_at = datetime.now(UTC).date()
                fetched.append(doc)
                print(f"ok      {doc.doc_id}  {len(content) // 1024} KB")
            except (httpx.HTTPError, OSError, ValueError) as exc:
                failures.append({"doc_id": doc.doc_id, "url": doc.url, "error": str(exc)})
                print(f"FAILED  {doc.doc_id}: {exc}")

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "fetched": [json.loads(d.model_dump_json()) for d in fetched],
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return fetched, failures


def load_manifest() -> list[SourceDoc]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No manifest. Run: biq fetch")
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [SourceDoc.model_validate(d) for d in data["fetched"]]
