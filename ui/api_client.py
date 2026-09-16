"""HTTP client for the Vervemint API. Both front ends (the Streamlit UI
and the Telegram bot) reach the backend only through this class, so
neither loads models or documents itself.

The API token can be a function (credentials.api_token): it is then read
on every request, so a token created or rotated later just works.

Every failure becomes an ApiError whose message can be shown to the
user as-is: the backend's own explanation when it answered, or how to
start it when it is not running.

Each call sends its own X-Request-ID, which the API reuses in its logs,
logs/llm.log and the trace, so one id follows a question end to end.
"""

import logging
import time
import uuid
from typing import Callable

import httpx

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """A request failed; the message is safe to show to the user."""


class ApiClient:
    def __init__(self, base_url: str,
                 token: str | None | Callable[[], str | None],
                 timeout: float):
        self.base_url = base_url
        self._token = token
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def _request(self, method: str, path: str,
                 llm_key: str | None = None, **kwargs):
        request_id = uuid.uuid4().hex[:12]
        headers = {"X-Request-ID": request_id}
        token = self._token() if callable(self._token) else self._token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # The provider key goes in a header, never in the JSON body, so
        # it cannot end up in body logs.
        if llm_key:
            headers["X-LLM-API-Key"] = llm_key
        # Page refreshes poll GET endpoints constantly; log those quietly.
        level = logging.DEBUG if method == "GET" else logging.INFO
        start = time.perf_counter()
        try:
            response = self._http.request(method, path, headers=headers,
                                          **kwargs)
        except httpx.TimeoutException as exc:
            logger.warning("%s %s timed out [%s]", method, path, request_id)
            raise ApiError("The backend took too long to answer. Try a "
                           "shorter question or a faster model.") from exc
        except httpx.TransportError as exc:
            logger.warning("%s %s unreachable [%s]", method, path, request_id)
            raise ApiError(
                f"Cannot reach the backend at {self.base_url}. Start it "
                "with: python -m uvicorn src.vervemint.api:app "
                "--host 127.0.0.1 --port 8000"
            ) from exc
        logger.log(level, "%s %s -> %d (%.0f ms) [%s]", method, path,
                   response.status_code,
                   (time.perf_counter() - start) * 1000, request_id)
        if response.status_code >= 400:
            raise ApiError(self._detail(response))
        return response.json() if response.content else None

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            detail = response.json().get("detail")
        except ValueError:
            return f"Backend error ({response.status_code})."
        if isinstance(detail, list):  # FastAPI validation errors
            return "; ".join(d.get("msg", str(d)) for d in detail)
        return str(detail or f"Backend error ({response.status_code}).")

    # --- Endpoints ----------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def assets(self) -> list[dict]:
        return self._request("GET", "/assets")

    def ollama_models(self) -> list[str]:
        return self._request("GET", "/providers/ollama/models")

    def connect(self, provider: str, model: str | None,
                key: str | None = None) -> dict:
        return self._request("POST", "/connect", llm_key=key,
                             json={"provider": provider, "model": model})

    def get_settings(self) -> dict:
        return self._request("GET", "/settings")

    def telegram_status(self) -> dict:
        """Small and cheap: the Settings page polls it."""
        return self._request("GET", "/telegram/status")

    def update_settings(self, changes: dict) -> dict:
        """Partial update; returns the new settings (see PUT /settings)."""
        return self._request("PUT", "/settings", json=changes)

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def upload(self, files: list[tuple[str, bytes]]) -> dict:
        return self._request(
            "POST", "/documents",
            files=[("files", (name, data)) for name, data in files],
        )

    def delete_document(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}")

    def ask(self, llm: dict, question: str, scope: dict) -> dict:
        return self._request(
            "POST", "/ask", llm_key=llm["key"],
            json={"question": question, "provider": llm["provider"],
                  "model": llm["model"], **scope},
        )

    def agent(self, llm: dict, question: str, scope: dict) -> dict:
        return self._request(
            "POST", "/agent", llm_key=llm["key"],
            json={"question": question, "provider": llm["provider"],
                  "model": llm["model"], **scope},
        )

    def analyze_log(self, llm: dict, question: str, scope: dict,
                    log_text: str, log_name: str) -> dict:
        return self._request(
            "POST", "/analyze-log", llm_key=llm["key"],
            json={"question": question, "provider": llm["provider"],
                  "model": llm["model"], "log_text": log_text,
                  "log_name": log_name, **scope},
        )

    def approve(self, draft: dict, approved_by: str) -> dict:
        return self._request("POST", "/work-orders",
                             json={**draft, "approved_by": approved_by})
