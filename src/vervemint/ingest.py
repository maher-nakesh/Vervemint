"""Turn raw documents into clean, metadata-tagged chunks.

Two sources of documents:
1. The built-in Panasonic library (parquet of OCR'd PDF pages).
2. Files a user uploads in the UI (PDF, TXT, MD).

Both go through the same `_chunk_page` function, so an uploaded manual
is chunked exactly like the library: images stripped, tables never cut
in half, and every chunk tagged with its source file and page so it can
be cited later.
"""

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from src.vervemint.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_UPLOAD_TYPES = ("pdf", "txt", "md")
# Bump when the chunking logic changes. It is part of the fingerprint of
# the library index and of every stored document, so both are rebuilt
# automatically instead of silently mixing old and new chunks.
CHUNKER_VERSION = 2

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image[^)]*\)")
_TABLE_RE = re.compile(r"<table.*?</table>", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str  # file the chunk came from, used in citations
    page_no: int
    category: str


def strip_images(md: str) -> str:
    """Replace embedded base64 images with a short placeholder.

    A placeholder instead of deletion keeps sentences like
    "see the figure below: [image]" readable.
    """
    return _IMAGE_RE.sub("[image]", md)


def _category_from_path(file_path: str) -> str:
    """Extract the library's category folder, e.g. 'technical_guides'."""
    match = re.search(r"/pdf/(?:category/)?([^/]+)/", file_path)
    return match.group(1) if match else "unknown"


def _split_preserving_tables(
    text: str, max_chars: int, overlap: int
) -> list[str]:
    """Split text into ~max_chars chunks, never cutting inside a table.

    Tables are emitted whole even if oversized: a truncated spec table
    is useless. Plain text is packed up to max_chars, carrying `overlap`
    characters into the next chunk so a sentence cut at a boundary keeps
    context on both sides.
    """
    segments: list[tuple[str, bool]] = []  # (text, is_table)
    last_end = 0
    for m in _TABLE_RE.finditer(text):
        if m.start() > last_end:
            segments.append((text[last_end:m.start()], False))
        segments.append((text[m.start():m.end()], True))
        last_end = m.end()
    if last_end < len(text):
        segments.append((text[last_end:], False))

    chunks: list[str] = []
    buffer = ""
    for seg_text, is_table in segments:
        if is_table:
            if buffer.strip():
                chunks.append(buffer)
                buffer = ""
            chunks.append(seg_text)
            continue

        start = 0
        while start < len(seg_text):
            room = max_chars - len(buffer)
            if room <= 0:
                chunks.append(buffer)
                buffer = buffer[-overlap:] if overlap else ""
                room = max_chars - len(buffer)
            buffer += seg_text[start:start + room]
            start += room

    if buffer.strip():
        chunks.append(buffer)

    return [c.strip() for c in chunks if c.strip()]


def page_header(clean_text: str, max_chars: int = 150) -> str:
    """The page's title lines, i.e. the text before its first table.

    On a datasheet this is e.g. "SPECIFICATION OF PANASONIC COMPRESSOR
    | Model: DJK51C73RAU". Pages that start with a table get no header
    rather than a header made of random table cells.
    """
    before_table = clean_text.split("<table", 1)[0]
    lines = [
        line.strip().lstrip("#").strip()
        for line in before_table.splitlines()
    ]
    lines = [line for line in lines if line and line != "[image]"]
    return " | ".join(lines)[:max_chars]


def _chunk_page(
    text: str, source: str, page_no: int, category: str
) -> list[Chunk]:
    """Clean and split one page of text into tagged chunks.

    Every chunk after the first is prefixed with the page's title. A
    chunk like "Approved oils: FREOL alpha10" is useless on its own:
    neither search nor the LLM can tell which product it belongs to,
    and identical spec rows from other datasheets look the same.
    """
    clean = strip_images(text)
    if not clean.strip():
        return []
    pieces = _split_preserving_tables(
        clean, settings.max_chunk_chars, settings.chunk_overlap_chars
    )
    header = page_header(clean)
    stem = Path(source).stem
    chunks = []
    for i, piece in enumerate(pieces):
        # The first chunk already starts with the title itself.
        if i > 0 and header:
            piece = f"[Page context: {header}]\n{piece}"
        chunks.append(
            Chunk(
                chunk_id=f"{stem}_p{page_no}_c{i}",
                text=piece,
                source=source,
                page_no=page_no,
                category=category,
            )
        )
    return chunks


# --- Built-in library ------------------------------------------------


def load_corpus() -> pd.DataFrame:
    """Read every parquet shard of the page-level library into one frame."""
    files = sorted(settings.corpus_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {settings.corpus_dir}"
        )
    frames = [
        pd.read_parquet(f, columns=["page_no", "file_path", "md_content"])
        for f in files
    ]
    return pd.concat(frames, ignore_index=True)


def build_chunks() -> list[Chunk]:
    """Load the whole library and return its chunks."""
    df = load_corpus()
    chunks: list[Chunk] = []
    for row in df.itertuples(index=False):
        chunks.extend(
            _chunk_page(
                row.md_content,
                source=Path(row.file_path).stem,
                page_no=int(row.page_no),
                category=_category_from_path(row.file_path),
            )
        )
    logger.info("Library: %d pages -> %d chunks", len(df), len(chunks))
    return chunks


# --- User uploads -------------------------------------------------------


def _read_pdf_pages(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [page.extract_text() or "" for page in reader.pages]


def chunks_from_upload(filename: str, data: bytes) -> list[Chunk]:
    """Turn one uploaded file into chunks.

    Page numbers start at 1 so citations match what the user sees in
    their PDF viewer. TXT/MD files are treated as a single page.
    Raises ValueError for unsupported or empty files.
    """
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix == "pdf":
        pages = _read_pdf_pages(data)
    elif suffix in ("txt", "md"):
        pages = [data.decode("utf-8", errors="replace")]
    else:
        raise ValueError(f"Unsupported file type: .{suffix}")

    chunks: list[Chunk] = []
    for page_no, text in enumerate(pages, start=1):
        chunks.extend(_chunk_page(text, filename, page_no, "uploaded"))

    if not chunks:
        # Typical cause: a scanned PDF with no text layer.
        raise ValueError(f"No extractable text in {filename}")
    logger.info(
        "Upload: %s -> %d pages, %d chunks", filename, len(pages), len(chunks)
    )
    return chunks
