"""The FastAPI backend end to end (real models and library index; the
LLM call is replaced by a canned answer so no model is needed)."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from src.vervemint import api as api_module
from src.vervemint import credentials
from src.vervemint.config import settings

MANUAL = b"Motor overheating: check the cooling fan and the ventilation.\n"
GEMINI_KEY = "AIzaTESTKEY1234567890abcdefghijklWx9Q"
BOT_TOKEN = "123456789:AAE1testTOKENtestTOKENtestTOKEN12"
NO_AUTH = {"Authorization": ""}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # A fresh credentials store: the lifespan creates the API token in it.
    # The in-process Telegram bot is off: tests never reach the network.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "data_dir", tmp_path_factory.mktemp("data"))
        mp.setattr(settings, "telegram_bot", False)
        with TestClient(api_module.app) as test_client:  # runs the lifespan
            test_client.headers["Authorization"] = (
                f"Bearer {credentials.api_token()}")
            yield test_client


@pytest.fixture
def temp_library(tmp_path, monkeypatch):
    """A throw-away index folder: a rename or a delete in a test must
    never reach the real data/index. The loaded index is put back, so
    the tests after this one can still search the library."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text(json.dumps({
        "fingerprint": "abc123", "n_chunks": 3,
        "built_at": "2026-09-14T13:13:18+00:00"}))
    monkeypatch.setattr(settings, "index_dir", index_dir)
    loaded = api_module._state["library"]
    yield index_dir
    api_module._state["library"] = loaded


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "documents_dir", tmp_path / "docs")
    api_module._doc_retrievers.clear()


def test_health_is_public_and_tagged(client):
    response = client.get("/health", headers=NO_AUTH)
    assert response.status_code == 200
    assert response.json()["library"]["available"]
    assert response.json()["auth_required"]
    assert response.headers["X-Request-ID"]


def test_generated_token_is_enforced(client, docs_dir):
    assert credentials.api_token()  # made on the first start
    assert client.get("/documents", headers=NO_AUTH).status_code == 401
    wrong = {"Authorization": "Bearer wrong"}
    assert client.get("/documents", headers=wrong).status_code == 401
    assert client.get("/documents").status_code == 200


def test_saved_key_comes_back_and_is_used(client):
    body = client.put("/settings", json={
        "keys": {"gemini": GEMINI_KEY}, "provider": "gemini",
        "models": {"gemini": "gemini-test"},
    }).json()
    # Returned so Settings can show and edit it (the caller holds the token).
    assert body["keys"]["gemini"] == {"value": GEMINI_KEY, "locked": False}
    assert body["provider"] == "gemini"
    # A request with no provider uses the choice made in the UI.
    llm = api_module._llm(api_module.LLMChoice(), None)
    assert (llm.provider, llm.model, llm.api_key) == (
        "gemini", "gemini-test", GEMINI_KEY)

    body = client.put("/settings", json={"keys": {"gemini": ""},
                                         "provider": "ollama"}).json()
    assert body["keys"]["gemini"]["value"] == ""
    assert body["provider"] == "ollama"


def test_environment_locks_a_setting(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFROMTHEENVIRONMENT0000")
    assert client.get("/settings").json()["keys"]["gemini"]["locked"]
    response = client.put("/settings", json={"keys": {"gemini": GEMINI_KEY}})
    assert response.status_code == 409
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_telegram_settings_are_validated(client):
    assert client.put("/settings", json={
        "telegram_bot_token": "not-a-token"}).status_code == 422
    assert client.put("/settings", json={
        "telegram_allowed_users": [-5]}).status_code == 422
    telegram = client.put("/settings", json={
        "telegram_bot_token": BOT_TOKEN,
        "telegram_allowed_users": [42, 7, 42],
    }).json()["telegram"]
    # Shown in full in Settings, so the user can check and edit it.
    assert telegram["token"] == {"value": BOT_TOKEN, "locked": False}
    assert telegram["allowed_users"] == [7, 42]
    assert credentials.telegram_bot_token() == BOT_TOKEN  # the bot reads it
    assert client.get("/telegram/status").json()["state"] == "not_running"
    client.put("/settings", json={"telegram_bot_token": "",
                                  "telegram_allowed_users": []})
    assert client.get("/settings").json()["telegram"]["token"]["value"] == ""


def test_ollama_address_is_editable(client):
    default = client.get("/settings").json()
    assert default["ollama_host"] == credentials.DEFAULT_OLLAMA_HOST
    assert default["ollama_host_locked"] is False
    assert client.put("/settings", json={
        "ollama_host": "127.0.0.1:11434"}).status_code == 422  # no scheme
    body = client.put("/settings", json={
        "ollama_host": "http://ollama:11434/"}).json()
    assert body["ollama_host"] == "http://ollama:11434"  # trailing / dropped
    assert credentials.ollama_host() == "http://ollama:11434"
    back = client.put("/settings", json={"ollama_host": ""}).json()
    assert back["ollama_host"] == credentials.DEFAULT_OLLAMA_HOST


def test_single_container_serves_the_ui_on_the_host_port():
    """One-container hosts route one port: the web UI takes it, the
    backend stays inside."""
    from src.vervemint import serve

    commands = serve.commands("1337")
    assert "--port=1337" in " ".join(commands["ui"]).replace(
        "--server.port=", "--port=")
    assert "--server.address=0.0.0.0" in commands["ui"]
    api = " ".join(commands["api"])
    assert "--host 127.0.0.1" in api and f"--port {serve.API_PORT}" in api


def test_backend_starts_the_telegram_bot(monkeypatch):
    """Starting the backend starts the bot: no second process to run."""
    started = []

    async def fake_supervise(stop):
        started.append(stop)
        await stop.wait()

    monkeypatch.setattr("ui.telegram_bot.supervise", fake_supervise)
    monkeypatch.setattr(settings, "telegram_bot", True)

    async def run() -> None:
        bot = await api_module._start_telegram_bot()
        assert bot is not None
        stop, task = bot
        await asyncio.sleep(0)  # let the task start
        assert started == [stop]
        stop.set()
        await task
        monkeypatch.setattr(settings, "telegram_bot", False)
        assert await api_module._start_telegram_bot() is None  # switched off

    asyncio.run(run())


def test_new_api_token_replaces_the_old_one(client):
    old = client.headers["Authorization"]
    body = client.put("/settings", json={"new_api_token": True}).json()
    client.headers["Authorization"] = f"Bearer {body['api_token']}"
    assert client.get("/settings", headers={"Authorization": old}
                      ).status_code == 401
    assert client.get("/settings").status_code == 200


def test_ui_password_can_be_generated_or_set(client):
    first = client.get("/settings").json()["ui_password"]
    generated = client.put("/settings", json={
        "new_ui_password": True}).json()["ui_password"]
    assert len(first) >= 16 and generated != first
    assert client.put("/settings", json={
        "ui_password": "short"}).status_code == 422
    assert client.put("/settings", json={
        "ui_password": "my-own-password"}).json()["ui_password"] == (
        "my-own-password")


def test_upload_is_cached_then_searchable(client, docs_dir, monkeypatch):
    files = [("files", ("manual.txt", MANUAL))]
    first = client.post("/documents", files=files).json()
    second = client.post("/documents", files=files).json()
    assert first["documents"][0]["cached"] is False
    assert second["documents"][0]["cached"] is True

    monkeypatch.setattr("src.vervemint.generate.chat",
                        lambda *args: "Check the cooling fan [S1].")
    doc_id = first["documents"][0]["doc_id"]
    body = client.post("/ask", json={
        "question": "What should I check if the motor overheats?",
        "scope": "documents", "document_ids": [doc_id],
    }).json()
    assert body["citations"] == ["manual.txt#p1"]
    assert "manual.txt#p1" in body["pages"]


def test_bad_uploads_are_reported_not_fatal(client, docs_dir, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    files = [("files", ("image.png", b"x")),
             ("files", ("big.txt", b"a" * (1024 * 1024 + 1)))]
    body = client.post("/documents", files=files).json()
    assert body["documents"] == [] and len(body["errors"]) == 2


def test_input_errors_get_clear_status_codes(client, docs_dir):
    assert client.post("/work-orders", json={
        "asset_id": "P-07", "priority": "urgent", "summary": "x",
        "approved_by": "me"}).status_code == 422
    assert client.delete("/documents/" + "0" * 32).status_code == 404
    assert client.post("/ask", json={
        "question": "x", "scope": "documents", "document_ids": []
    }).status_code == 400


def test_pasted_key_is_blocked(client):
    body = client.post("/ask", json={"question": "AIza" + "C" * 35}).json()
    assert body["blocked_reason"] == "secret_detected"


def test_log_analysis_endpoint(client, monkeypatch):
    monkeypatch.setattr("src.vervemint.generate.chat",
                        lambda *args: "Supply voltage 99 V is too low [S1].")
    response = client.post("/analyze-log", json={
        "log_text": "# Equipment: compressor DJK51C73RAU\n"
                    "2026-09-10 14:20:00 WARN  Supply voltage 99 V\n",
        "log_name": "cr2.log",
    })
    body = response.json()
    assert body["steps"][0]["arguments"]["query"].endswith("99 V")
    assert body["citations"] and body["pages"]
    # One id across the access log, pipeline log, llm.log and trace.
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_document_can_be_renamed(client, docs_dir, monkeypatch):
    """Renaming keeps the stored chunks and vectors; answers made after
    it cite the new name, so no cached retriever keeps the old one."""
    monkeypatch.setattr("src.vervemint.generate.chat",
                        lambda *args: "Check the cooling fan [S1].")
    doc_id = client.post("/documents", files=[
        ("files", ("manual.txt", MANUAL))]).json()["documents"][0]["doc_id"]

    def citations() -> list[str]:
        return client.post("/ask", json={
            "question": "What should I check if the motor overheats?",
            "scope": "documents", "document_ids": [doc_id],
        }).json()["citations"]

    assert citations() == ["manual.txt#p1"]  # caches a retriever
    body = client.patch(f"/documents/{doc_id}",
                        json={"name": "Pump manual"}).json()
    assert body["filename"] == "Pump manual.txt"  # the file type is kept
    assert body["chunks"] == client.get("/documents").json()[0]["chunks"]
    assert citations() == ["Pump manual.txt#p1"]

    assert client.patch(f"/documents/{doc_id}",
                        json={"name": "../etc"}).status_code == 400
    assert client.patch("/documents/" + "0" * 32,
                        json={"name": "x.txt"}).status_code == 404


def test_library_is_listed_renamed_and_deleted(client, temp_library):
    """The built-in index is one more source the UI can show, rename and
    remove, next to the user's own documents."""
    body = client.get("/library").json()
    assert body["available"]  # loaded, so it can be searched
    assert body["name"] == settings.corpus_dir.name  # until it is renamed
    assert body["built_at"] == "2026-09-14T13:13:18+00:00"

    named = client.patch("/library", json={"name": "Panasonic manuals"})
    assert named.json()["name"] == "Panasonic manuals"
    manifest = json.loads((temp_library / "manifest.json").read_text())
    assert manifest["name"] == "Panasonic manuals"
    assert manifest["fingerprint"] == "abc123"  # a rename rebuilds nothing
    assert client.patch("/library", json={"name": "a/b"}).status_code == 400

    assert client.delete("/library").status_code == 204
    assert not temp_library.exists()
    assert client.get("/library").json()["available"] is False
    assert client.delete("/library").status_code == 404
