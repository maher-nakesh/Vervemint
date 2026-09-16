"""Turn API answers into Telegram messages. Pure functions with no
Telegram import, so they are easy to test.

Telegram shows at most 4096 characters per message and understands only
a small HTML subset (<b>, <i>, <code>, ...). LLM answers are Markdown and
can contain '<' ("voltage < 103 V"), so text is HTML-escaped first and
only bold, headings, bullets and inline code are converted.
"""

import html
import re
from pathlib import Path

from ui.citations import build_entries

MAX_MESSAGE_CHARS = 4000  # Telegram's limit is 4096; keep a margin
LOG_EXTENSIONS = {".log", ".csv", ".txt"}
DOCUMENT_EXTENSIONS = {".pdf", ".md"}

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$", re.MULTILINE)
_BULLET_RE = re.compile(r"^(\s*)[*-]\s+", re.MULTILINE)
_CODE_RE = re.compile(r"`([^`\n]+)`")
_TAG_RE = re.compile(r"<[^>]+>")


def markdown_to_html(text: str) -> str:
    text = html.escape(text, quote=False)
    text = _CODE_RE.sub(r"<code>\1</code>", text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    # "### **Problem**" -> one bold line, not nested bold tags.
    text = _HEADING_RE.sub(
        lambda m: f"<b>{_TAG_RE.sub('', m.group(1))}</b>", text
    )
    return _BULLET_RE.sub(r"\1• ", text)


def plain_text(html_text: str) -> str:
    """Fallback when Telegram rejects the HTML (e.g. a tag cut in two)."""
    return html.unescape(_TAG_RE.sub("", html_text))


def footer(result: dict, provider: str, seconds: float) -> str:
    """One line under the answer, like the web UI's caption."""
    if result.get("blocked_reason"):
        return "Blocked by the input guardrail"
    parts = []
    if "stopped_reason" in result:
        parts.append("agent")
    elif "steps" in result:
        parts.append("log analysis")
    parts += [provider, f"{seconds:.1f}s"]
    if result.get("abstained"):
        parts.append("not enough evidence")
    stopped = result.get("stopped_reason")
    if stopped and stopped != "answered":
        parts.append(stopped.replace("_", " "))
    return " · ".join(parts)


def format_answer(result: dict, meta: str) -> str:
    """The answer with [1]-style citations, the cited sources, and the
    footer line, as Telegram HTML."""
    text, entries = build_entries(result["answer"],
                                  result.get("sources", []),
                                  result.get("pages", {}))
    parts = [markdown_to_html(text)]
    cited = [e for e in entries if e["number"]]
    if cited:
        lines = []
        for e in cited:
            where = f"{e['source']}, page {e['page_no']}"
            title = e["title"] if e["title"] == where else (
                f"{e['title']} ({where})")
            lines.append(f"[{e['number']}] {html.escape(title)}")
        parts.append("<b>Sources</b>\n" + "\n".join(lines))
    parts.append(f"<i>{html.escape(meta)}</i>")
    return "\n\n".join(parts)


def format_work_order(draft: dict) -> str:
    return (
        "<b>Work order awaiting your approval</b>\n"
        f"Asset <b>{html.escape(draft['asset_id'])}</b>, priority "
        f"<b>{html.escape(draft['priority'])}</b>\n"
        f"{html.escape(draft['summary'])}"
    )


def _pieces(paragraph: str, limit: int) -> list[str]:
    """A paragraph cut into parts of at most `limit` characters, at line
    breaks where possible."""
    if len(paragraph) <= limit:
        return [paragraph]
    pieces, current = [], ""
    for line in paragraph.split("\n"):
        while len(line) > limit:  # one line longer than a message
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split at paragraph boundaries, so a bold or code span (always
    within one line) is never cut in two."""
    messages, current = [], ""
    for paragraph in text.split("\n\n"):
        for piece in _pieces(paragraph, limit):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) > limit:
                messages.append(current)
                current = piece
            else:
                current = candidate
    if current:
        messages.append(current)
    return messages


def route_document(filename: str) -> str | None:
    """'log' for a machine log to diagnose, 'document' for a manual to
    store, None for anything else."""
    suffix = Path(filename).suffix.lower()
    if suffix in LOG_EXTENSIONS:
        return "log"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    return None
