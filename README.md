# BabyInfoQuick

BabyInfoQuick is a local portfolio demo for retrieval over pediatric guidance. Type a question, select an age band or let the app infer it, and inspect the passages returned from CDC, NICHD and USDA WIC PDFs. With a provider key, the app can also generate a cited answer from those passages.

## Benchmark

Five configurations were each run on the same 100 answerable questions. Thirty-one questions are age-sensitive and 69 are age-agnostic. The set combines 12 revised seed questions with 88 model-authored additions checked against the source PDFs. It has not had independent human adjudication. Each point of overall recall represents one question.

| Config | Recall@5 | MRR | Age-sensitive R@5 | Age-agnostic R@5 | Answer accuracy | p95 retrieval | Cost/query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense_fixed | 0.640 | 0.437 | 0.677 | 0.623 | not run | 106 ms | not measured |
| dense_section | 0.650 | 0.477 | 0.613 | 0.667 | not run | 105 ms | not measured |
| hybrid | 0.630 | 0.425 | 0.548 | 0.667 | not run | 112 ms | not measured |
| hybrid_rerank | 0.710 | 0.437 | 0.710 | 0.710 | not run | 8812 ms | not measured |
| hybrid_rerank_no_age_filter | 0.630 | 0.382 | 0.452 | 0.710 | not run | 8224 ms | not measured |

**Dense section retrieval is the demo default.** It put a labeled passage in the top five for 65 questions, with 105 ms p95 retrieval latency on this CPU. Hybrid reranking reached 71 questions at 8.8 seconds p95. Dense section retrieval had the highest MRR on this set.

**Age filtering made its largest difference on the age-sensitive subset.** With the same reranker, top-five hits rose from 14 to 22 of 31 when filtering was enabled. Among the 33 questions with an explicit age label, all returned passages overlapped the requested metadata in 33 cases with filtering and 9 without it. These are retrieval measurements. Answer quality, end-to-end generation latency and API cost were unmeasured because the run used no provider key.

## Run the local demo

For a publicly accessible instance, follow the [Linux hosting guide](docs/hosting.md). The hosted stack runs the app, Qdrant and an HTTPS reverse proxy together.

On this configured checkout, start Docker Desktop, then run these commands from the project directory in PowerShell.

```powershell
docker compose up -d
.\.venv\Scripts\biq.exe serve
```

Open [the local demo](http://127.0.0.1:8000). Try “How do I thaw frozen breast milk?” and expand a passage to inspect its text. The source link opens the PDF at the start page of the chunk. Try “What can my 9 month old do?” to see which age band the retriever selected. The comparison table on the page shows the recorded benchmark.

### First-time setup

Install Python 3.11 or newer and Docker with Compose. Run the following from the project root in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
docker compose up -d
biq fetch
biq parse --no-ocr
biq index --config hybrid
biq index --config dense_fixed
biq serve
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`. The initial build downloads the public PDFs and model weights. The four section-based configurations share one Qdrant collection and BM25 index. The fixed-window configuration uses its own collection. Indexing replaces the selected collection. For the text-only benchmark, use the parse command shown above. To enable OCR for a separate run, install Tesseract, run `biq parse` and rebuild both indexes.

### Answer generation

Without a provider key, the demo shows ranked source passages with their PDF links. To enable generated answers, set either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the project's `.env` file or environment and restart the app. With the bundled configurations an OpenAI key selects `gpt-4o-mini` and an Anthropic key selects `claude-opus-5`. If both are present, Anthropic is the default. Set `generation.provider` in a retrieval YAML to choose explicitly. Answer generation and evaluation with `--judge` use paid API calls.

```bash
biq configs
biq ask "How do I thaw frozen breast milk?" --config hybrid --show-chunks
biq ask "What can my 9 month old do?" --age infant_7_12
biq eval --check-labels
biq eval
python -m pytest -q
```

`POST /api/ask` accepts JSON with a question, an optional configuration name and optional age bands. The interactive schema and examples of the response fields are available in the [local API docs](http://127.0.0.1:8000/docs).

## What the benchmark compares

Fixed windows and section-aware chunks carry source, PDF page range and age-band metadata. At query time, the app resolves age from the selection or question and applies a metadata filter during vector search and before lexical top-k selection. Hybrid retrieval combines dense and BM25 ranks with reciprocal rank fusion. The optional cross-encoder scores up to 20 candidates before the final cutoff. A fifth configuration repeats reranking with age filtering disabled.

Qdrant stores the vectors. The BM25 index, parsed PDFs, chunks and per-query diagnostic logs are local under `data/`. Interactive query logs include question text and component scores. Evaluation traces identify which labeled passages each configuration retrieved. See the [evaluation protocol](eval/README.md), [recorded reports and traces](eval/benchmark/) and [failure analysis](docs/failure-analysis.md).

## Scope

This is an informational retrieval experiment. The source snapshot is dated August 2026. Generated answers carry a disclaimer and require resolvable inline passage citations. The prompt calls for refusal on diagnoses, medication dosages and emergency instructions. Empty retrieval skips the generator. Clinical accuracy and claim-level faithfulness have not been independently assessed.
