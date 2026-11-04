What this is

A retrieval-augmented question answering system over public pediatric and infant-care guidance. A parent asks a real question in plain language and gets an answer grounded in cited passages from official sources, filtered to the right age band.

Why it exists

This is a portfolio project. The point is not the chatbot — anyone can build a chatbot. The point is the retrieval engineering and the evaluation harness that proves it works.

Fourteen job postings in my current search listed RAG, vector search, or retrieval evaluation as a requirement. This project exists to turn that from MISSING to MEETS.

The artifact that matters is a table with real numbers in it:

Config	Recall@5	MRR	Answer accuracy	p95 latency	Cost/query
Dense only, fixed chunks
Dense only, section chunks
Hybrid (BM25 + dense)
Hybrid + cross-encoder rerank

If that table is empty at the end, the project failed regardless of how well the demo works.

Stack

Deliberately small. Do not expand it without asking.

Python 3.11+
FastAPI for serving
Qdrant (docker) for vectors — pgvector is an acceptable substitute
BAAI/bge-base-en-v1.5 via sentence-transformers for embeddings
rank_bm25 for lexical retrieval
BAAI/bge-reranker-base cross-encoder for reranking
pdfplumber / PyMuPDF for parsing, pytesseract fallback for scanned docs
Pydantic for structured output
Anthropic and OpenAI clients, provider-agnostic behind one interface
Docker

Do not add React, Next.js, Kubernetes, Celery, Redis, Terraform, or a frontend framework. A single server-rendered results page or a Streamlit view is sufficient. Time spent on infrastructure is time not spent on retrieval quality, which is the whole point.

Architecture

PDF corpus
  └─> parse + section-aware chunk
       └─> embed (BGE) ──> Qdrant  (payload: source, age_band, doc_type, url, page)
       └─> BM25 index
                │
user question   │
  └─> extract age_band from query (or take it from the UI)
       └─> metadata filter on age_band  ← filter BEFORE ranking, not after
            └─> hybrid retrieve (BM25 + dense), union
                 └─> cross-encoder rerank top-k
                      └─> LLM answer with inline citations
                           └─> structured response (Pydantic)

The one interesting retrieval problem

Guidance for a 3-week-old is different from guidance for a 9-month-old, and returning the wrong age band is the primary failure mode. Filter on age metadata before vector similarity, not after. Embeddings do not reliably separate "newborn" from "toddler" guidance because the surrounding language is nearly identical.

This is the design decision to be able to explain in an interview, and it should be measured: report recall separately for age-sensitive vs age-agnostic queries.

Corpus

Public, free, legally usable. Target 100–150 documents.

CDC — developmental milestones, infant nutrition, sleep safety, vaccine schedules
AAP / HealthyChildren.org — published guidance documents
NIH / NICHD — infant health publications
WHO — infant feeding and growth standards
Children's hospital patient handouts and discharge instructions (widely published as PDFs)
USDA WIC and state health department newborn care guides

Store the source URL and retrieval date in the payload for every chunk. Citations must resolve back to a real document.

Evaluation — do not skip this

Weekend 2 is hand-labeling 100 query→correct-passage pairs. It is boring and it is the entire reason this project is worth more than a demo.

Metrics to implement:

Retrieval: recall@1, recall@5, recall@10, MRR, nDCG
Answer: faithfulness (does every claim trace to a retrieved passage), citation precision, age-band correctness
Operational: p95 retrieval latency, p95 end-to-end latency, tokens and cost per query

Run every config in the ablation table against the same labeled set. Report all of them, including the ones that lost.

Also write one page of failure analysis: which queries still fail, why, what you'd do next. Interviewers trust the person who can name their system's weaknesses.

Conventions

Match the existing code style in the repo. Same naming, same structure, same general feel.
Do not add comments to code unless explicitly asked.
No emoji in code, commits, or output.
Small commits with plain descriptive messages.
Every retrieval config must be reproducible from a config file, not hardcoded, so ablations can be rerun.
Log every query with its retrieved chunks and scores. That log is the debugging surface.

Build order

Weekend 1 — scrape and parse the corpus, section-aware chunking, embed, index in Qdrant, basic dense retrieval, LLM answer with citations. Working end to end.
Weekend 2 — hand-label 100 query→passage pairs. Build the eval runner. Get baseline numbers for dense-only.
Weekend 3 — add BM25, add the cross-encoder reranker, add age-band metadata filtering. Run all four configs. Fill in the table.
Weekend 4 — Docker, deploy, README written around the numbers rather than the architecture diagram. Failure analysis page.

Safety and framing

This is a personal learning project, not a health product.
Every answer carries a plain disclaimer: informational only, not medical advice, contact a pediatrician for anything specific or urgent.
Never produce a diagnosis, a dosage, or an emergency instruction.
If retrieval returns nothing above the relevance threshold, say so and stop. Do not let the model answer from parametric knowledge — an ungrounded answer is the failure mode this whole system exists to prevent.
Do not market it publicly as a health tool.

Out of scope

Accounts, payments, mobile apps, feeding and sleep trackers, milestone reminders, anything multi-tenant. If it doesn't improve retrieval quality or the ability to measure it, it doesn't belong in this repo.
