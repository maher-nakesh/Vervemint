"""Hybrid retrieval: BM25 + dense search, fused with Reciprocal Rank
Fusion (RRF), then refined with a cross-encoder reranker.

Three stages, each fixing a weakness of the one before it:

1. BM25 finds exact keyword/part-number matches. Dense search finds
   paraphrases and semantic matches. Neither alone is enough (see the
   module docstring in index.py for why).
2. RRF combines the two ranked lists into one, using only rank position
   (not raw scores, which are on incomparable scales between BM25 and
   cosine similarity) -- this is the standard, parameter-light way to
   fuse rankings from different retrieval methods.
3. A cross-encoder reranker then re-scores the fused candidates. A
   bi-encoder (the embedding model used for dense search) encodes the
   query and each chunk *separately* and compares vectors -- fast, but
   it never lets the query and chunk attend to each other. A
   cross-encoder feeds (query, chunk) through the model *together*, so
   it directly answers "how relevant is this chunk to this exact
   query" -- slower (can't be precomputed/indexed) but much more
   accurate. That's why it only runs on the ~20-30 fused candidates,
   not the full 24k-chunk corpus.
"""

from dataclasses import dataclass

import bm25s
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.copilot.config import settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_pdf: str
    page_no: int
    category: str
    score: float  # final cross-encoder relevance score


class Retriever:
    """Loads all index artifacts once; call `.search()` per query.

    Loading models/indexes is expensive (seconds), searching is cheap
    (milliseconds) -- so this class is built to be instantiated once
    per process (e.g. once at API startup) and reused across requests.
    """

    def __init__(self) -> None:
        self._chunks = pd.read_parquet(settings.index_dir / "chunks.parquet")
        self._bm25 = bm25s.BM25.load(str(settings.index_dir / "bm25"))
        self._dense_index = faiss.read_index(
            str(settings.index_dir / "dense.faiss")
        )
        self._embedder = SentenceTransformer(settings.embedding_model)
        self._reranker = CrossEncoder(settings.reranker_model)

    def _bm25_search(self, query: str, k: int) -> list[int]:
        """Return chunk row-indices ranked by BM25, best first."""
        tokenized = bm25s.tokenize(
            [query], stopwords="en", show_progress=False
        )
        results, _scores = self._bm25.retrieve(
            tokenized, k=min(k, len(self._chunks)), show_progress=False
        )
        return results[0].tolist()

    def _dense_search(self, query: str, k: int) -> list[int]:
        """Return chunk row-indices ranked by cosine similarity, best first."""
        query_vec = self._embedder.encode(
            [query], normalize_embeddings=True
        )
        query_vec = np.asarray(query_vec, dtype="float32")
        _distances, indices = self._dense_index.search(query_vec, k)
        return indices[0].tolist()

    def _fuse_rrf(self, ranked_lists: list[list[int]]) -> list[int]:
        """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)).

        A chunk that appears near the top of *either* list gets a high
        score; a chunk appearing in both lists (even at moderate rank
        in each) can outrank a chunk that is #1 in only one list. This
        is what makes fusion better than picking either method alone.
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (
                    settings.rrf_k + rank
                )
        return sorted(scores, key=scores.get, reverse=True)

    def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Run the full hybrid pipeline and return the top_k results."""
        top_k = top_k or settings.rerank_top_k

        bm25_hits = self._bm25_search(query, settings.bm25_top_k)
        dense_hits = self._dense_search(query, settings.dense_top_k)
        fused = self._fuse_rrf([bm25_hits, dense_hits])

        candidates = self._chunks.iloc[fused]
        pairs = [(query, text) for text in candidates["text"]]
        rerank_scores = self._reranker.predict(pairs)

        ranked_order = np.argsort(rerank_scores)[::-1][:top_k]

        results = []
        for i in ranked_order:
            row = candidates.iloc[int(i)]
            results.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    source_pdf=row["source_pdf"],
                    page_no=int(row["page_no"]),
                    category=row["category"],
                    score=float(rerank_scores[i]),
                )
            )
        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "What communication protocol do digital air pressure sensors use?"
    for hit in retriever.search(query):
        preview = hit.text[:120]
        print(f"[{hit.score:.3f}] {hit.source_pdf} p{hit.page_no}")
        print(f"    {preview!r}")
