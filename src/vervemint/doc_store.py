"""Persistent store for uploaded documents: each file is chunked and
embedded once, then reused across uploads, sessions and restarts.

Every document lives in data/documents/<doc_id>/, where doc_id is the
SHA-256 of the file's bytes:
    original.<ext>   the uploaded file (needed to reprocess it later)
    chunks.parquet   chunk texts + metadata
    embeddings.npy   one vector per chunk
    meta.json        name, counts, date, processing fingerprint

- Uploading the same bytes again, under any file name, is a cache hit:
  nothing is re-read, re-chunked or re-embedded.
- If the chunking settings, the embedding model or the chunker version
  change, the stored fingerprint no longer matches and the document is
  reprocessed from original.<ext> the next time it is used.
- meta.json is written last, so an interrupted write is never mistaken
  for a finished document.
- Renaming only rewrites the name in meta.json and in the chunks, so a
  document keeps its vectors: nothing is embedded again.
"""

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.vervemint.config import settings
from src.vervemint.index import chunks_to_frame, clean_name, embed_texts
from src.vervemint.ingest import CHUNKER_VERSION, chunks_from_upload

logger = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"[0-9a-f]{32}")


def processing_fingerprint() -> str:
    """Everything that changes a document's chunks or vectors."""
    parts = [
        CHUNKER_VERSION,
        settings.max_chunk_chars,
        settings.chunk_overlap_chars,
        settings.embedding_model,
    ]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def _folder(doc_id: str) -> Path:
    # doc_id comes from API requests: only accept our own hex ids, so a
    # value like "../../x" can never turn into a file path.
    if not _DOC_ID_RE.fullmatch(doc_id):
        raise KeyError(doc_id)
    return settings.documents_dir / doc_id


def _read_meta(doc_id: str) -> dict | None:
    path = _folder(doc_id) / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _process(doc_id: str, filename: str, data: bytes,
             embedder: SentenceTransformer) -> dict:
    """Chunk + embed one file and save everything to its folder."""
    chunks = chunks_from_upload(filename, data)  # ValueError if unusable
    frame = chunks_to_frame(chunks)
    vectors = embed_texts(frame["text"].tolist(), embedder)

    folder = _folder(doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"original{Path(filename).suffix.lower()}").write_bytes(data)
    frame.to_parquet(folder / "chunks.parquet")
    np.save(folder / "embeddings.npy", vectors)
    meta = {
        "doc_id": doc_id,
        "filename": filename,
        "pages": int(frame["page_no"].nunique()),
        "chunks": len(frame),
        "size_bytes": len(data),
        "added_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": processing_fingerprint(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2),
                                      encoding="utf-8")
    logger.info("Document processed | %s -> %d chunks (%s)",
                filename, len(frame), doc_id)
    return meta


def add_document(filename: str, data: bytes,
                 embedder: SentenceTransformer) -> tuple[dict, bool]:
    """Store an uploaded file. Returns (metadata, was_already_stored).
    Raises ValueError for unsupported or text-less files."""
    doc_id = hashlib.sha256(data).hexdigest()[:32]
    meta = _read_meta(doc_id)
    if meta and meta["fingerprint"] == processing_fingerprint():
        logger.info("Document cache hit | %s (%s)", filename, doc_id)
        return meta, True
    return _process(doc_id, filename, data, embedder), False


def list_documents() -> list[dict]:
    """All stored documents, newest first."""
    if not settings.documents_dir.exists():
        return []
    metas = []
    for folder in settings.documents_dir.iterdir():
        meta = _read_meta(folder.name) if _DOC_ID_RE.fullmatch(
            folder.name) else None
        if meta:
            metas.append(meta)
    return sorted(metas, key=lambda m: m["added_at"], reverse=True)


def _clean_name(name: str, suffix: str) -> str:
    """A name the user typed, with the document's own file type kept:
    the suffix decides how the file is read if it is ever reprocessed."""
    name = name.strip()
    if suffix and name.lower().endswith(suffix.lower()):
        name = name[: -len(suffix)]
    return clean_name(name) + suffix


def rename_document(doc_id: str, name: str) -> dict | None:
    """Give a stored document another name, in its metadata and in its
    chunks, so citations show the new name. The chunks and vectors stay
    as they are: nothing is re-read, re-chunked or re-embedded.

    Returns the new metadata, or None when the id is unknown. Raises
    ValueError for an unusable name.
    """
    meta = _read_meta(doc_id)
    if meta is None:
        return None
    old_name = meta["filename"]
    new_name = _clean_name(name, Path(old_name).suffix)
    if new_name == old_name:
        return meta

    folder = _folder(doc_id)
    frame = pd.read_parquet(folder / "chunks.parquet")
    # chunk_id is "<stem>_p<page>_c<i>" (ingest.py): swapping the stem
    # gives exactly the ids a reprocessing under the new name would.
    cut = len(Path(old_name).stem)
    new_stem = Path(new_name).stem
    frame["chunk_id"] = [new_stem + chunk_id[cut:]
                         for chunk_id in frame["chunk_id"]]
    frame["source"] = new_name
    frame.to_parquet(folder / "chunks.parquet")
    meta["filename"] = new_name
    (folder / "meta.json").write_text(json.dumps(meta, indent=2),
                                      encoding="utf-8")
    logger.info("Document renamed | %s -> %s (%s)", old_name, new_name, doc_id)
    return meta


def delete_document(doc_id: str) -> bool:
    folder = _folder(doc_id)
    if not folder.exists():
        return False
    shutil.rmtree(folder)
    logger.info("Document deleted | %s", doc_id)
    return True


def load_documents(doc_ids: list[str], embedder: SentenceTransformer
                   ) -> tuple[pd.DataFrame, np.ndarray]:
    """Chunks and vectors of several documents, ready for
    Retriever.from_embeddings. Raises KeyError for an unknown id."""
    frames, vectors = [], []
    for doc_id in doc_ids:
        meta = _read_meta(doc_id)
        if meta is None:
            raise KeyError(doc_id)
        folder = _folder(doc_id)
        if meta["fingerprint"] != processing_fingerprint():
            logger.info("Settings changed, reprocessing %s", meta["filename"])
            original = next(folder.glob("original.*"))
            _process(doc_id, meta["filename"], original.read_bytes(), embedder)
        frames.append(pd.read_parquet(folder / "chunks.parquet"))
        vectors.append(np.load(folder / "embeddings.npy"))
    return pd.concat(frames, ignore_index=True), np.vstack(vectors)
