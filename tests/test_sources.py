"""UI source formatting: citation numbering, page grouping, tables."""

import pandas as pd

from ui.sources import _clean_table, build_entries


def _page(source: str, page_no: int) -> list[dict]:
    """A page as the API returns it: every chunk in reading order."""
    return [
        {"chunk_id": f"{source}_p{page_no}_c0",
         "text": "# Pump P1\n\nIntro text"},
        {"chunk_id": f"{source}_p{page_no}_c1",
         "text": "[Page context: Pump P1]\nOil: VG10"},
    ]


PAGES = {"a#p0": _page("a", 0), "b#p3": _page("b", 3)}


def _hit(chunk_id: str, source: str, page: int, score: float) -> dict:
    return {"chunk_id": chunk_id, "source": source, "page_no": page,
            "score": score}


def test_labels_become_numbers_and_pages_are_grouped():
    hits = [_hit("a_p0_c1", "a", 0, 0.9), _hit("a_p0_c0", "a", 0, 0.5),
            _hit("b_p3_c1", "b", 3, 0.4)]
    text, entries = build_entries(
        "Oil is VG10 [a#p0][a#p0]. Also [b#p3].", hits, PAGES
    )
    assert text == "Oil is VG10 [1]. Also [2]."
    assert [(e["label"], e["number"]) for e in entries] == [
        ("a#p0", 1), ("b#p3", 2)
    ]
    assert entries[0]["title"] == "Pump P1 · Intro text"
    assert entries[0]["score"] == 0.9
    # Both chunks of page a#p0 were read; only c1 of page b#p3 was.
    assert [read for _, read in entries[0]["page"]] == [True, True]
    assert [read for _, read in entries[1]["page"]] == [False, True]


def test_uncited_pages_are_kept_but_unnumbered():
    text, entries = build_entries(
        "Not in the documents.", [_hit("a_p0_c0", "a", 0, 0.3)], PAGES
    )
    assert text == "Not in the documents."
    assert entries[0]["number"] is None


def test_table_headers_are_flattened_and_unique():
    df = pd.DataFrame(
        [[1, 2, 3]],
        columns=pd.MultiIndex.from_tuples(
            [("Oil", "cm3"), ("Oil", "cm3"), ("Unnamed: 2", "in3")]
        ),
    )
    assert list(_clean_table(df).columns) == ["Oil cm3", "Oil cm3 (2)", "in3"]
