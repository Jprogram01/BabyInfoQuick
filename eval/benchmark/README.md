# Recorded benchmark

The November 2, 2026 local CPU run evaluated five configurations against the same 100 questions. The reports and ranked traces here are a saved snapshot. Running `biq eval` writes a new comparison under `eval/results/`.

The corpus was fetched on August 7, 2026. Both chunk sets were reproduced from the parsed documents, and every chunk payload was compared with the indexed Qdrant payload. The fixed index has 771 chunks and the section index has 1,125. The corpus manifest records the source URL, SHA-256 and indexed chunk counts for each document. The benchmark used text extraction without OCR.

The run used Windows, Python 3.12.0, CPU PyTorch 2.13.0 (8 inference threads), sentence-transformers 5.7.0, transformers 5.14.1, NumPy 2.5.1, rank-bm25 0.2.2, PyMuPDF 1.28.2 and qdrant-client 1.19.0. The Qdrant service ran locally. Model loading is outside the retrieval timer. The computer was also used for development during this run, so latencies are indicative.

| Model | Cached model revision |
| --- | --- |
| BAAI/bge-base-en-v1.5 | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| BAAI/bge-reranker-base | `2cfc18c9415c912f9d8155881c133215df768a70` |

Relevance was scored from the saved rankings against the final label set. nDCG uses the number of relevant chunks in the full index. The runtime, labels and evaluation procedure are shared across configurations.
