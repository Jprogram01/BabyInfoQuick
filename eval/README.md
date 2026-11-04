# Evaluation

The retrieval benchmark contains 100 answerable questions. The first 12 came from the original seed set. The other 88 were authored with model assistance against the local PDF text and checked against the referenced pages. They have not had independent human adjudication. Thirty-one questions are marked age-sensitive. The milestone questions deliberately ask about similar skills at neighboring ages.

## Reproduce the run

From the repository root, fetch and parse the documents, then build both indexes as described in the main README. Use `biq parse --no-ocr` to reproduce the text-only corpus used for the recorded run. The source PDF digests are in [corpus.json](benchmark/corpus.json).

```bash
biq eval --check-labels
biq eval --config hybrid
biq eval
```

The first command validates unique query IDs, PDF pages and normalized snippets against the parsed documents. It needs neither Qdrant nor a model key. The second evaluates one configuration. The last evaluates all five, including both cross-encoder runs, which can take tens of minutes on CPU.

Fresh results go to `eval/results/`. The saved [reports](benchmark/reports.json) include overall and subgroup metrics. The five ranked traces alongside them identify the chunks and relevance decisions behind each score.

## Passage labels

Each line in `labels.jsonl` is one query. A label can accept multiple passages. Snippets are independent of chunk boundaries, so the same labels can score both indexes.

```json
{"query_id":"q003","question":"When should I stop swaddling my baby?","age_bands":["infant_1_3"],"age_sensitive":true,"gold":[{"doc_id":"nichd_safe_sleep_brochure_2022","page":22,"snippet":"Stop swaddling baby when they start rolling over"}]}
```

A snippet match requires the same document and a contiguous match after normalization of case, punctuation and whitespace. For these labels the page is a source locator. A page-only label accepts any chunk from that document covering that PDF page. All page numbers are one-based PDF pages, which may differ from printed page numbers. Unlabeled alternative passages can still answer a question.

Explicit age bands go to the retriever. When a label omits them the normal question parser runs. The age-filter-off configuration disables filtering for both retrieval branches.

## Metrics

| Metric | How it is calculated |
| --- | --- |
| Recall@1/5/10 | Fraction of questions with at least one labeled passage among the first k chunks (hit rate). |
| MRR | Mean reciprocal rank of the first matching chunk among the ten evaluated. A miss scores zero. |
| nDCG@10 | Binary relevance with logarithmic rank discount. The ideal ranking uses all matching chunks in the corresponding index, including any the retriever missed. |
| Age-band correctness | Among queries with explicit bands, fraction with a nonempty top five where every chunk overlaps a requested band. Measures metadata consistency. |
| p95 retrieval | Nearest-rank percentile over all 100 calls. Includes query encoding, retrieval, fusion and reranking where enabled. Excludes model loading and gold-label scoring. |

## Optional answer evaluation

```bash
biq eval --config hybrid --judge
```

Generation and judging use the same selected provider and model. Either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is sufficient with the bundled configurations. Both steps use paid API calls. Estimates of per-answer cost use the rates in `babyinfoquick/answer/llm.py` and exclude evaluator charges. An unmeasured metric is stored as `null`.

The judge grades accuracy and faithfulness against the retrieved context. It can credit a refusal when the context lacks the answer. Citation precision counts cited sources matching the labeled passages. Four extra questions in `refusal_controls.jsonl` exercise requests outside the corpus or outside the demo's scope. Their refusal rate is reported separately from the 100 answerable questions. The saved retrieval run did not call either provider.
