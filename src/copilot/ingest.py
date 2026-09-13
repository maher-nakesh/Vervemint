"""Turn the raw Panasonic corpus into clean, metadata-tagged chunks.

The corpus ships as per-page markdown extracted by OCR from PDFs. Two
problems make it unusable for retrieval as-is:

1. Pages embed base64-encoded images inline (`![](data:image/png;...)`),
   which can be tens of thousands of characters of noise per page.
2. A "page" is not a good retrieval unit: some pages hold three unrelated
   spec tables, others are a single table that continues from the
   previous page. Splitting blindly by character count can cut a table
   in half, which destroys the one thing an engineer actually needs.

This module produces a flat list of `Chunk` objects, each carrying the
text plus enough metadata (source pdf, page, category) to cite it later.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.copilot.config import settings

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image[^)]*\)")
_TABLE_RE = re.compile(r"<table.*?</table>", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_pdf: str
    page_no: int
    category: str


def _strip_images(md: str) -> str:
    """Replace embedded base64 images with a short placeholder.

    We keep a placeholder instead of deleting the match outright so a
    line like "See figure below: ![]()" doesn't collapse into a
    dangling sentence that confuses the embedding model.
    """
    return _IMAGE_RE.sub("[image]", md)


def _category_from_path(file_path: str) -> str:
    """Extract the dataset's category folder, e.g. 'technical_guides'.

    The raw `file_path` column looks like
    '/home/parsa/panasonic/pdf/category/technical_guides/pdf_file_123.pdf'.
    This is the only place that string layout is parsed, so if the
    dataset ever changes shape, this is the one function to fix.
    """
    match = re.search(r"/pdf/(?:category/)?([^/]+)/", file_path)
    return match.group(1) if match else "unknown"


def _split_preserving_tables(
    text: str, max_chars: int, overlap: int
) -> list[str]:
    """Split text into chunks of roughly `max_chars`, without ever
    cutting inside a `<table>...</table>` block.

    Strategy: cut the text into a sequence of (table, non-table)
    segments first. Tables are always emitted whole, even if that makes
    one chunk larger than `max_chars` -- a truncated spec table is
    useless, so we accept an oversized-but-intact chunk instead.
    Non-table segments are packed together up to `max_chars`, with a
    small character overlap between consecutive chunks so a sentence
    split across a boundary still has context on both sides.
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

        # Pack plain text into the buffer, flushing whenever it would
        # overflow max_chars.
        start = 0
        while start < len(seg_text):
            room = max_chars - len(buffer)
            if room <= 0:
                chunks.append(buffer)
                buffer = buffer[-overlap:] if overlap else ""
                room = max_chars - len(buffer)
            piece = seg_text[start:start + room]
            buffer += piece
            start += room

    if buffer.strip():
        chunks.append(buffer)

    return [c.strip() for c in chunks if c.strip()]


def load_corpus() -> pd.DataFrame:
    """Read every parquet shard of the page-level corpus into one frame.

    Returns columns: page_no, file_path, md_content (raw, uncleaned).
    """
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
    """Run the full ingestion pipeline: load -> clean -> chunk -> tag.

    This is the single entry point `index.py` calls. Keeping one
    function as the "front door" of the module means the internal
    helpers above are free to change without breaking callers.
    """
    df = load_corpus()
    chunks: list[Chunk] = []

    for row in df.itertuples(index=False):
        clean_text = _strip_images(row.md_content)
        if not clean_text.strip():
            continue

        category = _category_from_path(row.file_path)
        pdf_name = Path(row.file_path).stem
        pieces = _split_preserving_tables(
            clean_text,
            settings.max_chunk_chars,
            settings.chunk_overlap_chars,
        )
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{pdf_name}_p{row.page_no}_c{i}",
                    text=piece,
                    source_pdf=pdf_name,
                    page_no=int(row.page_no),
                    category=category,
                )
            )

    return chunks


if __name__ == "__main__":
    result = build_chunks()
    print(f"Built {len(result)} chunks from corpus at {settings.corpus_dir}")
    print("Example chunk:")
    print(result[0])
