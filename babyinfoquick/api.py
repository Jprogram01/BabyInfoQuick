from __future__ import annotations

import json
import math
import time
from collections import deque
from functools import lru_cache
from threading import Lock

import httpx
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from babyinfoquick.age import ALL_BANDS, BAND_LABELS, AgeBand
from babyinfoquick.answer.generate import DISCLAIMER
from babyinfoquick.config import BM25_DIR, EVAL_DIR, REPO_ROOT, list_configs, load_config, settings
from babyinfoquick.service import ask
from babyinfoquick.types import Answer

app = FastAPI(title="BabyInfoQuick", description="Grounded QA over public pediatric guidance")
templates = Jinja2Templates(directory=str(REPO_ROOT / "templates"))
_ask_times: deque[float] = deque()
_ask_lock = Lock()


class AskRequest(BaseModel):
    question: str
    config: str | None = None
    age_bands: list[AgeBand] | None = None


def _validate(question: str, config: str) -> None:
    if not question.strip():
        raise HTTPException(422, "Enter a question.")
    if config not in list_configs():
        raise HTTPException(422, "Choose one of the listed retrieval configurations.")
    if settings.max_asks_per_minute:
        with _ask_lock:
            now = time.monotonic()
            while _ask_times and _ask_times[0] <= now - 60:
                _ask_times.popleft()
            if len(_ask_times) >= settings.max_asks_per_minute:
                retry = math.ceil(_ask_times[0] + 60 - now)
                raise HTTPException(429, "Demo is busy. Try again shortly.",
                                    headers={"Retry-After": str(retry)})
            _ask_times.append(now)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "configs": list_configs()}


@app.get("/ready")
def ready() -> dict:
    configs = [load_config(name) for name in list_configs()]
    for cfg in configs:
        if cfg.retrieval.lexical and not (BM25_DIR / f"{cfg.collection}.pkl").is_file():
            raise HTTPException(503, "Lexical index is missing. Run corpus setup.")
    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
    try:
        with httpx.Client(base_url=settings.qdrant_url, headers=headers, timeout=2) as client:
            for name in sorted({cfg.collection for cfg in configs if cfg.retrieval.dense}):
                response = client.get(f"collections/{name}")
                response.raise_for_status()
                if not response.json()["result"]["points_count"]:
                    raise HTTPException(503, "Vector index is empty. Run corpus setup.")
    except httpx.HTTPError as exc:
        raise HTTPException(503, "Vector index is unavailable. Check Qdrant and corpus setup.") from exc
    return {"status": "ready"}


@app.post("/api/ask", response_model=Answer)
def api_ask(request: AskRequest) -> Answer:
    _validate(request.question, request.config or settings.default_config)
    answer, _ = ask(
        request.question,
        request.config or settings.default_config,
        request.age_bands,
    )
    return answer


@app.get("/", response_class=HTMLResponse)
def index(request: Request, config: str = Query(default="")) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=_context(config=config or settings.default_config),
    )


@app.post("/", response_class=HTMLResponse)
def submit(
    request: Request,
    question: str = Form(...),
    config: str = Form(default=""),
    age_band: str = Form(default=""),
) -> HTMLResponse:
    config_name = config or settings.default_config
    if age_band and age_band not in ALL_BANDS:
        raise HTTPException(422, "Choose one of the listed age bands.")
    _validate(question, config_name)
    bands = [AgeBand(age_band)] if age_band else None
    answer, result = ask(question, config_name, bands)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=_context(
            config=config_name,
            question=question,
            age_band=age_band,
            answer=answer,
            result=result,
        ),
    )


@lru_cache(maxsize=1)
def _benchmark() -> list[dict]:
    return json.loads((EVAL_DIR / "benchmark" / "reports.json").read_text(encoding="utf-8"))


def _context(config: str, **extra) -> dict:
    context = {
        "configs": list_configs(),
        "config": config,
        "bands": [(b.value, BAND_LABELS[b]) for b in ALL_BANDS],
        "question": "",
        "age_band": "",
        "answer": None,
        "result": None,
        "benchmark": _benchmark(),
        "disclaimer": DISCLAIMER,
    }
    context.update(extra)
    return context
