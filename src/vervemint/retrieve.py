"""Hybrid retrieval: BM25 + dense search, fused with Reciprocal Rank
Fusion (RRF), then refined with a cross-encoder reranker.

1. BM25 finds exact keyword/part-number matches; dense search finds
   paraphrases. Neither alone is enough.
2. RRF merges the two ranked lists using rank position only -- BM25
   scores and cosine similarities are on incomparable scales.
3. A cross-encoder reads (query, chunk) together and scores relevance
   directly. It is too slow for the whole corpus, so it only reranks
   the ~40 fused candidates.

A Retriever can be built three ways:
- `Retriever.from_disk(...)`       -> the saved built-in library index
- `Retriever.from_embeddings(...)` -> stored documents whose vectors are
                                      already on disk (no re-embedding)
- `Retriever.from_chunks(...)`     -> any chunks, embedded on the spot
"""

import logging
import time
from dataclasses import dataclass

import bm25s
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.vervemint.config import settings
from src.vervemint.index import (
    GPU_LOCK,
    build_bm25,
    chunks_to_frame,
    embed_texts,
    faiss_index,
    release_gpu_cache,
)
from src.vervemint.ingest import Chunk
from src.vervemint.logging_config import quiet_model_loading

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page_no: int
    category: str
    score: float  # cross-encoder relevance score


def load_models() -> tuple[SentenceTransformer, CrossEncoder]:
    """Load the embedder and reranker. Slow (seconds): do it once per
    process and share the models across every Retriever.

    On a GPU they run in half precision: half the GPU memory, which
    leaves room for a local Ollama model on the same card, and ~3x
    faster reranking. Measured on 150 evaluation questions: the same
    top passage for 99% of them, and no abstain decision changed.
    """
    quiet_model_loading()
    start = time.perf_counter()
    on_gpu = torch.cuda.is_available()
    kwargs = {"model_kwargs": {"dtype": torch.float16}} if on_gpu else {}
    embedder = SentenceTransformer(settings.embedding_model, **kwargs)
    reranker = CrossEncoder(settings.reranker_model, **kwargs)
    logger.info(
        "Models loaded on %s (%s) in %.1fs", embedder.device,
        "float16" if on_gpu else "float32", time.perf_counter() - start,
    )
    return embedder, reranker


class Retriever:
    def __init__(
        self,
        chunks: pd.DataFrame,
        bm25: bm25s.BM25,
        dense_index: faiss.Index,
        embedder: SentenceTransformer,
        reranker: CrossEncoder,
    ) -> None:
        self._chunks = chunks
        self._bm25 = bm25
        self._dense_index = dense_index
        self._embedder = embedder
        self._reranker = reranker

    @property
    def size(self) -> int:
        return len(self._chunks)

    @classmethod
    def from_disk(
        cls, embedder: SentenceTransformer, reranker: CrossEncoder
    ) -> "Retriever":
        """Load the built-in library index written by index.py."""
        chunks = pd.read_parquet(settings.index_dir / "chunks.parquet")
        bm25 = bm25s.BM25.load(str(settings.index_dir / "bm25"))
        dense = faiss.read_index(str(settings.index_dir / "dense.faiss"))
        logger.info("Library index loaded: %d chunks", len(chunks))
        return cls(chunks, bm25, dense, embedder, reranker)

    @classmethod
    def from_embeddings(
        cls,
        chunks: pd.DataFrame,
        embeddings: np.ndarray,
        embedder: SentenceTransformer,
        reranker: CrossEncoder,
    ) -> "Retriever":
        """Index chunks whose vectors were computed earlier. Only BM25
        and FAISS are rebuilt, which takes milliseconds."""
        start = time.perf_counter()
        chunks = chunks.reset_index(drop=True)
        retriever = cls(
            chunks, build_bm25(chunks["text"].tolist()),
            faiss_index(embeddings), embedder, reranker,
        )
        logger.info("In-memory index built: %d chunks in %.2fs",
                    len(chunks), time.perf_counter() - start)
        return retriever

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        embedder: SentenceTransformer,
        reranker: CrossEncoder,
    ) -> "Retriever":
        """Embed and index chunks in memory."""
        df = chunks_to_frame(chunks)
        vectors = embed_texts(df["text"].tolist(), embedder)
        return cls.from_embeddings(df, vectors, embedder, reranker)

    def location(self, row: int) -> tuple[str, int]:
        """(source, page_no) of a chunk row -- the unit used for citing
        and for scoring retrieval in the evaluation scripts."""
        chunk = self._chunks.iloc[row]
        return chunk["source"], int(chunk["page_no"])

    def page_chunks(self, source: str, page_no: int) -> list[tuple[str, str]]:
        """(chunk_id, text) for every chunk of one page, in reading order.
        Used by the UI to show a cited passage inside its full page."""
        rows = self._chunks[
            (self._chunks["source"] == source)
            & (self._chunks["page_no"] == page_no)
        ]
        return list(zip(rows["chunk_id"], rows["text"]))

    # The four stages are public so evaluation/eval_retrieval.py can
    # score each one separately. `search()` chains all of them.

    def bm25_search(self, query: str, k: int) -> list[int]:
        """Chunk row positions ranked by BM25, best first."""
        tokenized = bm25s.tokenize(
            [query], stopwords="en", show_progress=False
        )
        results, _scores = self._bm25.retrieve(
            tokenized, k=min(k, self.size), show_progress=False
        )
        return results[0].tolist()

    def dense_search(self, query: str, k: int) -> list[int]:
        """Chunk row positions ranked by cosine similarity, best first."""
        query_vec = self._embedder.encode(
            [query], normalize_embeddings=True
        )
        query_vec = np.asarray(query_vec, dtype="float32")
        # Asking FAISS for more rows than it holds returns -1
        # placeholders, which would silently point at the last chunk.
        _distances, indices = self._dense_index.search(
            query_vec, min(k, self.size)
        )
        return indices[0].tolist()

    @staticmethod
    def fuse_rrf(ranked_lists: list[list[int]]) -> list[int]:
        """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)).

        A chunk ranked well by both methods beats a chunk that is #1 in
        only one of them.
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (
                    settings.rrf_k + rank
                )
        return sorted(scores, key=scores.get, reverse=True)

    def rerank(
        self, query: str, rows: list[int]
    ) -> list[tuple[int, float]]:
        """Score (query, chunk) pairs with the cross-encoder.
        Returns (row, score) pairs, best first."""
        texts = self._chunks["text"].iloc[rows].tolist()
        scores = self._reranker.predict([(query, t) for t in texts])
        order = np.argsort(scores)[::-1]
        return [(rows[i], float(scores[i])) for i in order]

    def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Hybrid search + rerank; returns the top_k best chunks."""
        top_k = top_k or settings.rerank_top_k
        with GPU_LOCK:
            fused = self.fuse_rrf([
                self.bm25_search(query, settings.bm25_top_k),
                self.dense_search(query, settings.dense_top_k),
            ])
            ranked = self.rerank(query, fused)[:top_k]
            release_gpu_cache()
        results = []
        for row, score in ranked:
            chunk = self._chunks.iloc[row]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    source=chunk["source"],
                    page_no=int(chunk["page_no"]),
                    category=chunk["category"],
                    score=score,
                )
            )
        return results
