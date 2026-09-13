"""Build and persist the two retrieval indexes: BM25 (keyword) and FAISS
(dense/semantic).

Why two indexes instead of one:
- Dense embeddings are good at *meaning* ("what stops a motor overheating"
  matches "cooling system maintenance" even with no shared words) but bad
  at *exact tokens* -- a part number like "ERJ-D1" or "SAM D21E" often
  gets embedded close to unrelated text because the embedding model was
  never trained to treat alphanumeric codes as meaningful units.
- BM25 is the opposite: unbeatable at exact keyword/part-number matches,
  useless at paraphrases.
- Running both and fusing the results (done in retrieve.py) covers both
  failure modes. This is the standard "hybrid search" pattern used in
  production RAG systems.

This module writes three artifacts to `settings.index_dir`:
  chunks.parquet   -- the chunk text + metadata (so retrieve.py doesn't
                       need to re-run ingestion)
  bm25/             -- the bm25s index directory
  dense.faiss       -- the FAISS index of embeddings
"""

import json

import bm25s
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.copilot.config import settings
from src.copilot.ingest import build_chunks


def _save_chunks_table(chunks) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "text": [c.text for c in chunks],
            "source_pdf": [c.source_pdf for c in chunks],
            "page_no": [c.page_no for c in chunks],
            "category": [c.category for c in chunks],
        }
    )
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.index_dir / "chunks.parquet")
    return df


def _build_bm25(texts: list[str]) -> None:
    """Tokenize and index all chunk texts with bm25s, then persist.

    bm25s pre-computes term statistics (document frequencies, lengths)
    at index time so that querying later is just a lookup, not a full
    re-scan of every document.
    """
    tokenized = bm25s.tokenize(texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokenized)
    retriever.save(str(settings.index_dir / "bm25"))


def _build_dense(texts: list[str]) -> None:
    """Embed every chunk and store the vectors in a flat FAISS index.

    We use IndexFlatIP (exact inner-product search) rather than an
    approximate index (HNSW/IVF). At ~24k chunks, exact search over
    384-dim vectors is a few milliseconds -- approximate indexing only
    pays off in the millions-of-vectors range, so adding it here would
    be complexity with no benefit, which is exactly the kind of
    over-engineering to avoid in a system this size.
    """
    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # required for inner product == cosine
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(settings.index_dir / "dense.faiss"))


def build_all() -> None:
    """Entry point: ingest -> save metadata -> build both indexes."""
    chunks = build_chunks()
    df = _save_chunks_table(chunks)
    texts = df["text"].tolist()

    print(f"Indexing {len(texts)} chunks...")
    _build_bm25(texts)
    _build_dense(texts)

    manifest = {
        "n_chunks": len(texts),
        "embedding_model": settings.embedding_model,
    }
    (settings.index_dir / "manifest.json").write_text(json.dumps(manifest))
    print(f"Done. Indexes written to {settings.index_dir}")


if __name__ == "__main__":
    build_all()
