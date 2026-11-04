# BabyInfoQuick

A small RAG app for questions about baby care. It searches PDFs from the CDC, NICHD, and USDA WIC, then shows the relevant text and links to the source pages. Add an OpenAI or Anthropic key and it can write an answer with citations.

I built this to get more hands-on experience with retrieval and evaluation. The main thing I wanted to test was age filtering. Advice for a newborn and a 9-month-old can sound similar, so the app uses age tags to narrow the search.

## Results

I compared five setups on 100 questions. “Found in top 5” means at least one labeled passage showed up in the first five results. On my CPU, 95% of searches finished within the time listed.

| Setup | Found in top 5 | MRR | p95 search time |
| --- | --- | --- | --- |
| Dense search, fixed chunks | 64% | 0.437 | 106 ms |
| Dense search, section chunks | 65% | 0.477 | 105 ms |
| Hybrid search | 63% | 0.425 | 112 ms |
| Hybrid + reranking | 71% | 0.437 | 8.8 s |
| Hybrid + reranking, age filter off | 63% | 0.382 | 8.2 s |

The app defaults to dense search with section chunks. Reranking found more of the labeled passages, but it was much slower. For the 31 questions where age mattered, turning the age filter on improved top-five hits from 14 to 22.

The test set started with 12 questions. The other 88 were AI-assisted and checked against the PDFs. They still need independent human review. These results measure retrieval. I haven't measured generated-answer accuracy or API cost.

The [evaluation notes](eval/README.md) explain the metrics. The [saved results](eval/benchmark/) and [failure analysis](docs/failure-analysis.md) show what worked and what still needs work.

## Run it

You'll need Python 3.11+ and Docker. Start Docker, then run this from the project folder in PowerShell.

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

On macOS or Linux, use `source .venv/bin/activate` to activate the environment.

The first setup downloads the PDFs and models, then builds both indexes. This takes a while on CPU. After that, you just need Docker running and `biq serve`.

Open [localhost:8000](http://127.0.0.1:8000). Try “How do I thaw frozen breast milk?” or “What can my 9 month old do?”

To get generated answers, put either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in a local `.env` file and restart the app. If both are set, it uses Anthropic. You can choose a provider and model in the YAML files under `configs/`. API calls cost money.

## Other commands

```bash
biq configs
biq ask "How do I thaw frozen breast milk?" --show-chunks
biq eval --check-labels
biq eval
python -m pytest -q
```

`biq eval --judge` also generates and grades answers using the selected provider. It needs an API key.

The API is available at `POST /api/ask`. You can try it in the [local API docs](http://127.0.0.1:8000/docs).

## Hosting

The [hosting guide](docs/hosting.md) covers running it on a Linux server with Docker. The package includes the app, Qdrant, and an HTTPS proxy.

This is a learning project. The source documents were downloaded in August 2026. It isn't medical advice.
