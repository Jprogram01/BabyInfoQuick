from __future__ import annotations

from babyinfoquick.config import PipelineConfig
from babyinfoquick.corpus.chunk import chunk_doc, read_chunks, write_chunks
from babyinfoquick.corpus.fetch import load_manifest
from babyinfoquick.corpus.parse import load_parsed
from babyinfoquick.index.bm25_store import BM25Store
from babyinfoquick.index.embedder import Embedder
from babyinfoquick.index.qdrant_store import QdrantStore
from babyinfoquick.types import Chunk


def build_chunks(config: PipelineConfig) -> list[Chunk]:
    docs = load_manifest()
    chunks: list[Chunk] = []
    for doc in docs:
        parsed = load_parsed(doc.doc_id)
        doc_chunks = chunk_doc(parsed, doc, config.chunking)
        print(f"chunked {doc.doc_id}  {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
    write_chunks(chunks, config.chunking.strategy)
    return chunks


def build_index(config: PipelineConfig, reuse_chunks: bool = False) -> int:
    if reuse_chunks:
        chunks = read_chunks(config.chunking.strategy)
        print(f"loaded {len(chunks)} chunks from disk")
    else:
        chunks = build_chunks(config)

    if not chunks:
        raise RuntimeError("No chunks produced. Run: biq fetch && biq parse")

    embedder = Embedder(config.embedding)
    print(f"embedding {len(chunks)} chunks with {config.embedding.model_name}")
    vectors = embedder.embed_passages([c.text for c in chunks], show_progress=True)

    store = QdrantStore(config.collection, config.embedding.dim)
    store.recreate()
    store.upsert(chunks, vectors)
    print(f"qdrant  collection={config.collection}  points={store.count()}")

    bm25 = BM25Store.build(chunks)
    bm25.save(config.collection)
    print(f"bm25    index={config.collection}  docs={len(chunks)}")

    return len(chunks)
