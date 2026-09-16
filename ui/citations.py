"""Citation numbering shared by the Streamlit UI and the Telegram bot.

The pipeline cites evidence as [source#pN] labels: exact, but made for
machines (the API and the traces keep them). build_entries turns them
into [1], [2] markers plus one entry per retrieved page, titled with the
page's heading instead of a file id. No Streamlit import: the bot uses
this module too.
"""

import re

from src.vervemint.ingest import page_header

_LABEL_RE = re.compile(r"\[([^\[\]]+?#p\d+)\]")
_REPEATED_MARKER_RE = re.compile(r"(\[\d+\])(?:\s*,?\s*\1)+")
_CONTEXT_RE = re.compile(r"^\[Page context: (.*?)\]\n", re.DOTALL)


def body(text: str) -> str:
    """Chunk text without the '[Page context: ...]' search prefix."""
    match = _CONTEXT_RE.match(text)
    return text[match.end():] if match else text


def _title(page: list[dict], source: str, page_no: int) -> str:
    header = page_header(body(page[0]["text"])) if page else ""
    if not header:
        return f"{source}, page {page_no}"
    # The first few heading lines name the product; the rest is body text.
    title = " · ".join(header.split(" | ")[:3])
    return title if len(title) <= 90 else title[:87] + "..."


def build_entries(
    answer_text: str, sources: list[dict], pages: dict[str, list[dict]]
) -> tuple[str, list[dict]]:
    """Number the answer's citations and describe every retrieved page.

    `sources` and `pages` come straight from the API's /ask or /agent
    response. Returns the answer text with [1]-style markers, and one
    entry per page: its number (None if not cited), title, score, and
    the page's chunks in reading order, flagged where the model read.
    """
    entries_by_label: dict[str, dict] = {}
    for src in sources:
        label = f"{src['source']}#p{src['page_no']}"
        entry = entries_by_label.setdefault(label, {
            "label": label,
            "source": src["source"],
            "page_no": src["page_no"],
            "score": src["score"],
            "read_ids": set(),
        })
        entry["score"] = max(entry["score"], src["score"])
        entry["read_ids"].add(src["chunk_id"])

    numbers: dict[str, int] = {}

    def _number(match: re.Match) -> str:
        label = match.group(1)
        if label not in entries_by_label:
            return match.group(0)
        numbers.setdefault(label, len(numbers) + 1)
        return f"[{numbers[label]}]"

    display_text = _LABEL_RE.sub(_number, answer_text)
    display_text = _REPEATED_MARKER_RE.sub(r"\1", display_text)

    entries = []
    for label, entry in entries_by_label.items():
        page = pages.get(label, [])
        read_ids = entry.pop("read_ids")
        entry["number"] = numbers.get(label)
        entry["title"] = _title(page, entry["source"], entry["page_no"])
        entry["page"] = [(c["text"], c["chunk_id"] in read_ids)
                         for c in page]
        entries.append(entry)
    entries.sort(key=lambda e: (e["number"] is None, e["number"] or 0,
                                -e["score"]))
    return display_text, entries
