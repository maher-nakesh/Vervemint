"""Build the two retrieval indexes over a list of chunks.

- BM25 (keyword): best at exact tokens such as part numbers ("ERJ-D1").
- FAISS (dense): best at meaning and paraphrases.
retrieve.py searches both and fuses the results.

Embedding is the expensive step, so it is separate from building the
FAISS index: stored documents (doc_store.py) keep their vectors on disk
and rebuild FAISS from them in milliseconds, without re-embedding.

The built-in library is indexed once and saved to settings.index_dir.
`build_all()` skips the work when nothing changed: the manifest records a
fingerprint of the dataset files and of every setting that affects the
chunks or vectors. Run with --force to rebuild anyway.
"""

import argparse
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone

import bm25s
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.vervemint.config import settings
from src.vervemint.ingest import CHUNKER_VERSION, Chunk, build_chunks
from src.vervemint.logging_config import quiet_model_loading

logger = logging.getLogger(__name__)

# One GPU, several request threads (FastAPI's thread pool): model
# inference runs one call at a time, so concurrent requests queue
# instead of running the GPU out of memory. LLM calls are not locked.
GPU_LOCK = threading.Lock()


def release_gpu_cache() -> None:
    """Give PyTorch's cached GPU memory back to the driver after a batch.
    PyTorch keeps it for reuse (about 1 GB after one rerank); on an 8 GB
    GPU that pushes a local Ollama model partly into system RAM, where it
    generates several times slower. Call while holding GPU_LOCK."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_INDEX_FILES = ("chunks.parquet", "bm25", "dense.faiss", "manifest.json")


def chunks_to_frame(chunks: list[Chunk]) -> pd.DataFrame:
    """Chunk list -> DataFrame; row position is the id both indexes use."""
    return pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "text": [c.text for c in chunks],
            "source": [c.source for c in chunks],
            "page_no": [c.page_no for c in chunks],
            "category": [c.category for c in chunks],
        }
    )


def build_bm25(texts: list[str]) -> bm25s.BM25:
    """Tokenize and index texts for keyword search."""
    tokenized = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=False)
    return retriever


def embed_texts(
    texts: list[str], embedder: SentenceTransformer
) -> np.ndarray:
    """Unit-length float32 vectors, so inner product equals cosine."""
    with GPU_LOCK:
        vectors = embedder.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        release_gpu_cache()
    return np.asarray(vectors, dtype="float32")


def faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Exact inner-product index: milliseconds at this scale; an
    approximate index only pays off at millions of vectors."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def library_fingerprint() -> str:
    """Changes whenever the library index would come out different."""
    files = sorted(settings.corpus_dir.glob("*.parquet"))
    parts = [
        [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files],
        CHUNKER_VERSION,
        settings.max_chunk_chars,
        settings.chunk_overlap_chars,
        settings.embedding_model,
    ]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def library_is_current() -> bool:
    """True if a complete index exists and was built from the same
    dataset files with the same settings."""
    if not all((settings.index_dir / f).exists() for f in _INDEX_FILES):
        return False
    manifest = json.loads((settings.index_dir / "manifest.json").read_text())
    return manifest.get("fingerprint") == library_fingerprint()


def build_all(force: bool = False) -> bool:
    """Index the built-in library. Returns False if it was already up to
    date and nothing was rebuilt."""
    if not force and library_is_current():
        logger.info("Library index is up to date; nothing to rebuild "
                    "(use --force to rebuild anyway)")
        return False

    chunks = build_chunks()
    df = chunks_to_frame(chunks)
    texts = df["text"].tolist()
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Indexing %d library chunks", len(texts))
    df.to_parquet(settings.index_dir / "chunks.parquet")
    build_bm25(texts).save(str(settings.index_dir / "bm25"))
    quiet_model_loading()
    embedder = SentenceTransformer(settings.embedding_model)
    faiss.write_index(
        faiss_index(embed_texts(texts, embedder)),
        str(settings.index_dir / "dense.faiss"),
    )

    # Written last: an interrupted build leaves no valid manifest, so the
    # next run rebuilds instead of trusting half-written files.
    manifest = {
        "fingerprint": library_fingerprint(),
        "n_chunks": len(texts),
        "embedding_model": settings.embedding_model,
        "chunker_version": CHUNKER_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (settings.index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    logger.info("Library index written to %s", settings.index_dir)
    return True


if __name__ == "__main__":
    from src.vervemint.logging_config import setup_logging

    parser = argparse.ArgumentParser(description="Build the library index.")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the index is up to date")
    setup_logging()
    build_all(force=parser.parse_args().force)
