"""Readable sources and clickable citations for the chat UI.

ui/citations.py numbers the answer's citations ([1], [2]) and builds one
entry per retrieved page. This module renders those entries:

- below the answer, one entry per cited page, titled with the page's
  heading instead of a file id
- clicking an entry opens the exact passage the model read, highlighted,
  followed by the full page for context; spec tables render as tables
"""

import re
from io import StringIO

import pandas as pd
import streamlit as st

from ui.citations import body, build_entries  # noqa: F401 (re-exported)

_TABLE_SPLIT_RE = re.compile(r"(<table.*?</table>)", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$")


# --- Rendering ------------------------------------------------------------


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the multi-row headers pandas builds from rowspan/colspan
    and make column names unique (the table widget requires it)."""
    if isinstance(df.columns, pd.MultiIndex):
        names = [
            " ".join(dict.fromkeys(
                str(p) for p in col if not str(p).startswith("Unnamed")
            ))
            for col in df.columns
        ]
    else:
        names = ["" if str(c).startswith("Unnamed") else str(c)
                 for c in df.columns]
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    df.columns = unique
    return df.fillna("").astype(str)


def _render_text(text: str) -> None:
    lines = []
    for line in text.splitlines():
        line = _TAG_RE.sub("", line).strip()
        if not line or line == "[image]":
            continue
        heading = _HEADING_RE.match(line)
        lines.append(f"**{heading.group(1)}**" if heading else line)
    if lines:
        st.markdown("\n\n".join(lines))


def render_passage(text: str) -> None:
    """Show one chunk readably: text as markdown, HTML tables as tables.
    Document HTML is never rendered as HTML (it could contain scripts)."""
    for part in _TABLE_SPLIT_RE.split(body(text)):
        if not part.strip():
            continue
        if part.startswith("<table"):
            try:
                table = pd.read_html(StringIO(part))[0]
                st.dataframe(_clean_table(table), hide_index=True)
                continue
            except ValueError:
                pass  # unparseable table: fall through to plain text
        _render_text(part)


@st.dialog("Source", width="large")
def show_source(entry: dict) -> None:
    st.markdown(f"#### {entry['title']}")
    st.caption(f"{entry['source']}, page {entry['page_no']} · "
               f"relevance {entry['score']:.2f}")
    st.markdown("**Passage the answer is based on**" if entry["number"]
                else "**Retrieved passage (not cited in the answer)**")
    for text, read in entry["page"]:
        if read:
            with st.container(border=True):
                render_passage(text)
    with st.expander("Full page, in reading order"):
        for text, read in entry["page"]:
            if read:
                st.caption("Passage the model read:")
                with st.container(border=True):
                    render_passage(text)
            else:
                render_passage(text)


def _entry_button(entry: dict, key: str) -> None:
    prefix = f"[{entry['number']}] " if entry["number"] else ""
    if st.button(prefix + entry["title"], key=f"{key}_{entry['label']}",
                 type="tertiary", icon=":material/description:"):
        show_source(entry)
    st.caption(f"{entry['source']}, page {entry['page_no']} · "
               f"relevance {entry['score']:.2f}")


def render_entries(entries: list[dict], key: str) -> None:
    """Cited pages first, then the rest of what retrieval returned."""
    cited = [e for e in entries if e["number"]]
    other = [e for e in entries if not e["number"]]
    if cited:
        st.markdown("**Sources** (click to open the passage)")
        for entry in cited:
            _entry_button(entry, key)
    if other:
        with st.expander(f"Also retrieved, not cited ({len(other)})"):
            for entry in other:
                _entry_button(entry, key)
