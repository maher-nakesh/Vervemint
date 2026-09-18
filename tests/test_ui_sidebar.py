"""The Chat sidebar, run headlessly with Streamlit's AppTest: one ticked
list of everything already indexed (the built-in index and the stored
documents), each row with a pen that opens rename / delete.

The backend is a stub, so this loads no models and touches no network."""

import pytest
from streamlit.testing.v1 import AppTest

LIBRARY = {"available": True, "name": "panasonic_v0_0", "chunks": 23737,
           "built_at": "2026-09-14T13:13:18+00:00"}
DOC = {"doc_id": "a" * 32, "filename": "pump.pdf", "chunks": 12,
       "added_at": "2026-09-18T08:00:00+00:00"}


class FakeClient:
    """Only what sidebar_sources calls."""

    def __init__(self):
        self.done = []

    def library(self):
        return dict(LIBRARY)

    def documents(self):
        return [dict(DOC)]

    def rename_library(self, name):
        self.done.append(("rename_library", name))
        return {**LIBRARY, "name": name}

    def delete_library(self):
        self.done.append(("delete_library",))

    def rename_document(self, doc_id, name):
        self.done.append(("rename_document", doc_id, name))
        return {**DOC, "filename": name}

    def delete_document(self, doc_id):
        self.done.append(("delete_document", doc_id))


def _script(client) -> None:
    """The sidebar on its own, as the Chat page runs it."""
    import streamlit as st

    import ui.app as app

    for name, default in [("upload_report", None), ("uploader_round", 0),
                          ("flash", []), ("editing", None),
                          ("edit_mode", app.EDIT_MENU), ("edit_round", 0)]:
        st.session_state.setdefault(name, default)
    st.text(repr(app.sidebar_sources(client)))  # what gets searched


@pytest.fixture
def app_test():
    client = FakeClient()
    test = AppTest.from_function(_script, args=(client,))
    test.client = client  # the test reads what the sidebar asked for
    return test.run()


def _row_labels(test) -> list[str]:
    return [box.label for box in test.sidebar.checkbox]


def test_every_indexed_source_is_listed_with_its_own_pen(app_test):
    assert _row_labels(app_test) == ["panasonic_v0_0 :gray[· 23,737]",
                                     "pump.pdf :gray[· 12]"]
    assert app_test.sidebar.checkbox(key="pick_library").help == (
        "23,737 passages · added 2026-09-14")
    # One pen per row, and nothing opened until it is pressed.
    assert [b.key for b in app_test.sidebar.button] == [
        "pen_library", f"pen_{DOC['doc_id']}"]
    assert not app_test.sidebar.text_input


def _scope(test) -> str:
    return test.main.text[0].value


def _question(test) -> str:
    """The line the delete confirmation asks."""
    return next(c.value for c in test.sidebar.caption if "Delete" in c.value)


def test_a_document_is_searched_by_default(app_test):
    # The uploaded document, not the whole built-in index.
    assert _scope(app_test) == (
        "{'scope': 'documents', 'document_ids': ['%s']}" % DOC["doc_id"])


def test_the_index_is_searched_on_its_own(app_test):
    picked = app_test.sidebar.checkbox(key="pick_library").check().run()
    assert picked.session_state.selected_sources == ["library"]
    assert _scope(picked) == "{'scope': 'library', 'document_ids': []}"
    # Ticking a document again drops the index: one scope per question.
    back = picked.sidebar.checkbox(key=f"pick_{DOC['doc_id']}").check().run()
    assert back.session_state.selected_sources == [DOC["doc_id"]]
    assert _scope(back) == (
        "{'scope': 'documents', 'document_ids': ['%s']}" % DOC["doc_id"])


def test_nothing_ticked_asks_for_a_source(app_test):
    empty = app_test.sidebar.checkbox(key=f"pick_{DOC['doc_id']}"
                                      ).uncheck().run()
    assert empty.session_state.selected_sources == []
    assert "Tick at least one source" in empty.sidebar.info[0].value


def test_the_pen_opens_rename_and_delete(app_test):
    opened = app_test.sidebar.button(key=f"pen_{DOC['doc_id']}").click().run()
    assert opened.session_state.editing == DOC["doc_id"]
    # The pen makes room for the two actions, on the same row.
    doc = DOC["doc_id"]
    assert [b.key for b in opened.sidebar.button] == [
        "pen_library", f"rename_{doc}", f"delete_{doc}", f"close_{doc}"]
    assert [b.help for b in opened.sidebar.button][1:] == [
        "Rename", "Delete", "Close"]
    assert not opened.sidebar.text_input  # nothing opened below yet
    closed = opened.sidebar.button(key=f"close_{doc}").click().run()
    assert closed.session_state.editing is None  # and back to the pen
    assert [b.key for b in closed.sidebar.button] == [
        "pen_library", f"pen_{doc}"]


def test_rename_turns_the_name_into_a_text_box(app_test):
    opened = app_test.sidebar.button(key=f"pen_{DOC['doc_id']}").click().run()
    typing = opened.sidebar.button(key=f"rename_{DOC['doc_id']}").click().run()
    box = typing.sidebar.text_input[0]
    assert box.value == "pump.pdf"  # the stored name, ready to type over
    # The row is the box now: its checkbox is gone while it is open.
    assert [c.key for c in typing.sidebar.checkbox] == ["pick_library"]
    assert [b.help for b in typing.sidebar.button][1:] == ["Save", "Cancel"]

    typed = box.set_value("cooling pump").run()
    assert app_test.client.done == []  # a form: nothing is sent while typing
    saved = typed.sidebar.button(key=f"save_{DOC['doc_id']}").click().run()
    assert app_test.client.done == [
        ("rename_document", DOC["doc_id"], "cooling pump")]
    assert saved.session_state.editing is None  # and the row goes back
    assert saved.sidebar.checkbox(key=f"pick_{DOC['doc_id']}"
                                  ).label.startswith("pump.pdf")


def test_rename_can_be_left_without_changing_anything(app_test):
    opened = app_test.sidebar.button(key=f"pen_{DOC['doc_id']}").click().run()
    typing = opened.sidebar.button(key=f"rename_{DOC['doc_id']}").click().run()
    typed = typing.sidebar.text_input[0].set_value("half typed").run()
    cancelled = typed.sidebar.button(key=f"cancel_{DOC['doc_id']}"
                                     ).click().run()
    assert app_test.client.done == []  # nothing was sent
    assert cancelled.session_state.editing is None
    # Opening it again starts from the stored name, not from the typing.
    again = cancelled.sidebar.button(key=f"pen_{DOC['doc_id']}").click().run()
    again = again.sidebar.button(key=f"rename_{DOC['doc_id']}").click().run()
    assert again.sidebar.text_input[0].value == "pump.pdf"


def test_delete_asks_first(app_test):
    opened = app_test.sidebar.button(key="pen_library").click().run()
    asking = opened.sidebar.button(key="delete_library").click().run()
    question = _question(asking)
    assert "**Delete panasonic_v0_0?**" in question
    assert "embedding its passages again" in question

    kept = asking.sidebar.button(key="no_library").click().run()
    assert app_test.client.done == []  # Cancel deletes nothing
    assert kept.session_state.editing is None

    opened = kept.sidebar.button(key="pen_library").click().run()
    asking = opened.sidebar.button(key="delete_library").click().run()
    gone = asking.sidebar.button(key="yes_library").click().run()
    assert app_test.client.done == [("delete_library",)]
    assert gone.session_state.editing is None


def test_a_document_is_deleted_after_the_same_question(app_test):
    opened = app_test.sidebar.button(key=f"pen_{DOC['doc_id']}").click().run()
    asking = opened.sidebar.button(key=f"delete_{DOC['doc_id']}").click().run()
    question = _question(asking)
    assert "**Delete pump.pdf?**" in question
    assert "uploading the file again" in question
    gone = asking.sidebar.button(key=f"yes_{DOC['doc_id']}").click().run()
    assert app_test.client.done == [("delete_document", DOC["doc_id"])]
    assert gone.session_state.editing is None
