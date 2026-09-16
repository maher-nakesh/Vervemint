"""Provider-agnostic LLM calls: Ollama (local), Claude, OpenAI, Gemini.

Two entry points, both provider-neutral:
- `chat(config, system, user)`: one question in, text out (RAG answers).
- `chat_with_tools(config, system, messages, tools)`: one agent step;
  the model either answers or asks to call tools.

Each provider has its own wire format for tools and tool results, so
those differences are contained here. agent.py never touches a
provider SDK directly.

Every call is logged twice: one summary line in logs/vervemint.log
(latency, tokens, Ollama's generation speed) and the full text sent and
received in logs/llm.log, both tagged with the API request id.
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
import httpx
import ollama
import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src.vervemint import credentials
from src.vervemint.config import settings
from src.vervemint.guardrails import redact_secrets
from src.vervemint.logging_config import LLM_IO_LOGGER
from src.vervemint.tracing import current_request_id

logger = logging.getLogger(__name__)
# Full prompts and replies; logging_config sends them to logs/llm.log.
_io_logger = logging.getLogger(LLM_IO_LOGGER)

Provider = Literal["ollama", "claude", "openai", "gemini"]
_PROVIDER_NAMES = {
    "ollama": "Ollama",
    "claude": "Claude",
    "openai": "OpenAI",
    "gemini": "Gemini",
}

# Claude models that support server-side refusal fallbacks.
_CLAUDE_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}


class LLMError(RuntimeError):
    """A provider call failed; the message is safe to show to the user."""


@dataclass(frozen=True)
class LLMConfig:
    provider: Provider
    model: str
    # repr=False keeps the key out of logs and error messages even if
    # someone logs the whole config object.
    api_key: str | None = field(default=None, repr=False)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolTurn:
    """One model step: final text, or tool calls to execute."""

    text: str
    tool_calls: list[ToolCall]
    # Provider-native assistant message (a dict, or a Gemini Content);
    # append to history unchanged.
    assistant_message: Any
    usage: dict[str, Any] = field(default_factory=dict)


@contextmanager
def _provider_errors(config: LLMConfig):
    """Translate every SDK exception into one readable LLMError."""
    name = _PROVIDER_NAMES[config.provider]
    try:
        yield
    except (anthropic.AuthenticationError, openai.AuthenticationError) as e:
        raise LLMError(f"Invalid {name} API key.") from e
    except (anthropic.NotFoundError, openai.NotFoundError) as e:
        raise LLMError(f"Unknown {name} model: {config.model}") from e
    except (anthropic.RateLimitError, openai.RateLimitError) as e:
        raise LLMError(f"{name} rate limit reached. Retry shortly.") from e
    except (anthropic.APIConnectionError, openai.APIConnectionError) as e:
        raise LLMError(f"Cannot reach the {name} API.") from e
    except (anthropic.APIStatusError, openai.APIStatusError) as e:
        raise LLMError(
            _status_message(name, e.status_code, _error_detail(e))
        ) from e
    except genai_errors.APIError as e:
        raise LLMError(_gemini_error_message(e, config.model)) from e
    except httpx.TransportError as e:
        # google-genai does not wrap network failures in its own errors.
        raise LLMError(f"Cannot reach the {name} API.") from e
    except ollama.ResponseError as e:
        raise LLMError(f"Ollama error: {e.error}") from e
    except ConnectionError as e:
        raise LLMError(
            "Cannot reach Ollama. Start it with `ollama serve`."
        ) from e


def _error_detail(e: Exception) -> str:
    """The provider's own explanation, e.g. 'The model is overloaded'."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(getattr(e, "message", "") or "")


def _status_message(name: str, code: int, detail: str) -> str:
    """Readable text for an HTTP error, including the provider's reason
    so the user can tell 'my request is wrong' from 'their side is down'."""
    detail = " ".join(detail.split())[:200]
    reason = f" Provider says: {detail}" if detail else ""
    if code >= 500:
        return (
            f"{name} is temporarily unavailable ({code}).{reason} This is "
            "on the provider's side: retry in a moment or pick another model."
        )
    return f"{name} API error ({code}).{reason}"


def _gemini_error_message(e: genai_errors.APIError, model: str) -> str:
    # Google returns 400 (not 401) for a bad key, so check the message.
    if e.code in (401, 403) or "API key" in (e.message or ""):
        return "Invalid Gemini API key."
    if e.code == 404:
        return f"Unknown Gemini model: {model}"
    if e.code == 429:
        return "Gemini rate limit reached. Retry shortly."
    return _status_message("Gemini", e.code, e.message or e.status or "")


def _claude_request(config: LLMConfig, **params) -> Any:
    client = anthropic.Anthropic(api_key=config.api_key)
    if config.model in _CLAUDE_FALLBACK_MODELS:
        # If the model declines, the API retries on a fallback model
        # inside the same call.
        params["betas"] = ["server-side-fallback-2026-07-01"]
        params["extra_body"] = {"fallbacks": "default"}
    return client.beta.messages.create(
        model=config.model, max_tokens=16000, **params
    )


def _ollama_options() -> dict[str, float | int]:
    return {"temperature": settings.ollama_temperature,
            "num_ctx": settings.ollama_num_ctx}


def _ollama() -> ollama.Client:
    """A client for the server address set in Settings, read per call so
    a change there applies without a restart."""
    return ollama.Client(host=credentials.ollama_host())


def ollama_models() -> list[str]:
    """The models that server has. Raises ConnectionError if it is down."""
    return [model.model for model in _ollama().list().models]


def _gemini_client(config: LLMConfig) -> genai.Client:
    # Retry temporary failures (overload, rate limit) with backoff, as
    # the Anthropic and OpenAI SDKs already do by default.
    return genai.Client(
        api_key=config.api_key,
        http_options=genai_types.HttpOptions(
            timeout=int(settings.gemini_timeout_s * 1000),  # milliseconds
            retry_options=genai_types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=8.0,  # fail within seconds, not minutes
                http_status_codes=[429, 500, 502, 503, 504],
            )
        ),
    )


def _gemini_thinking() -> genai_types.ThinkingConfig:
    return genai_types.ThinkingConfig(
        thinking_level=settings.gemini_thinking_level.upper()
    )


def _gemini_text(content: genai_types.Content | None) -> str:
    """Visible answer text; skips the model's internal thinking parts."""
    if content is None:
        return ""
    return "".join(
        p.text for p in (content.parts or []) if p.text and not p.thought
    )


def _claude_text(response: Any) -> str:
    if response.stop_reason == "refusal":
        return "The model declined to answer this request."
    # Content is a list of blocks (thinking, text, tool_use, ...).
    return "".join(b.text for b in response.content if b.type == "text")


# --- Logging ------------------------------------------------------------


def _usage(provider: str, response: Any) -> dict[str, Any]:
    """Token counts from a provider response. Ollama also reports its
    generation speed: a sharp drop means the model no longer fits in GPU
    memory (see retrieval_device in config.yaml)."""
    if provider == "ollama":
        count = getattr(response, "eval_count", None)
        ns = getattr(response, "eval_duration", None)
        load_ns = getattr(response, "load_duration", None)
        usage = {
            "prompt_tokens": getattr(response, "prompt_eval_count", None),
            "output_tokens": count,
            "tokens_per_s": round(count / (ns / 1e9), 1)
            if count and ns else None,
            "load_ms": round(load_ns / 1e6) if load_ns else None,
        }
    elif provider == "gemini":
        u = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": getattr(u, "prompt_token_count", None),
            "output_tokens": getattr(u, "candidates_token_count", None),
            "thinking_tokens": getattr(u, "thoughts_token_count", None),
        }
    elif provider == "claude":
        u = getattr(response, "usage", None)
        usage = {"prompt_tokens": getattr(u, "input_tokens", None),
                 "output_tokens": getattr(u, "output_tokens", None)}
    else:
        u = getattr(response, "usage", None)
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                 "output_tokens": getattr(u, "completion_tokens", None)}
    return {k: v for k, v in usage.items() if v is not None}


def _jsonable(obj: Any) -> Any:
    """json.dumps fallback for SDK message objects (Gemini Content,
    Claude content blocks), which are pydantic models."""
    if isinstance(obj, bytes):  # e.g. Gemini thought signatures
        return f"<{len(obj)} bytes>"
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return repr(obj)


def _render(messages: list[Any]) -> str:
    """Plain text messages as they are, anything structured as JSON."""
    blocks = []
    for m in messages:
        if (isinstance(m, dict) and set(m) == {"role", "content"}
                and isinstance(m["content"], str)):
            blocks.append(f"----- {m['role']} -----\n{m['content']}")
            continue
        role = (m.get("role") if isinstance(m, dict)
                else getattr(m, "role", None))
        body = json.dumps(m, default=_jsonable, ensure_ascii=False, indent=1)
        blocks.append(f"----- {role or 'message'} -----\n{body}")
    return redact_secrets("\n".join(blocks))


def _log_request(config: LLMConfig, kind: str, messages: list[Any],
                 tools: list[dict[str, Any]] | None = None) -> None:
    if not settings.log_llm_messages:
        return
    names = f" | tools: {', '.join(t['name'] for t in tools)}" if tools else ""
    _io_logger.info("[%s] >>> %s / %s (%s)%s\n%s",
                    current_request_id() or "-", config.provider,
                    config.model, kind, names, _render(messages))


def _log_reply(config: LLMConfig, what: str, start: float,
               usage: dict[str, Any], text: str,
               calls: list[ToolCall] | None = None) -> None:
    request_id = current_request_id() or "-"
    ms = (time.perf_counter() - start) * 1000
    stats = " ".join(f"{k}={v}" for k, v in usage.items())
    called = (f" tools_called={[c.name for c in calls]}"
              if calls is not None else "")
    logger.info("[%s] %s | provider=%s model=%s%s latency_ms=%.0f | %s",
                request_id, what, config.provider, config.model, called, ms,
                stats)
    if not settings.log_llm_messages:
        return
    body = f"----- reply -----\n{text}"
    if calls:
        body += "\n----- tool calls -----\n" + json.dumps(
            [{"name": c.name, "arguments": c.arguments} for c in calls],
            default=_jsonable, ensure_ascii=False,
        )
    _io_logger.info("[%s] <<< %s / %s | latency_ms=%.0f %s\n%s", request_id,
                    config.provider, config.model, ms, stats,
                    redact_secrets(body))


def _log_failure(config: LLMConfig, what: str, start: float,
                 exc: LLMError) -> None:
    request_id = current_request_id() or "-"
    ms = (time.perf_counter() - start) * 1000
    logger.error("[%s] %s failed | %s | latency_ms=%.0f | %s", request_id,
                 what, config, ms, exc)
    if settings.log_llm_messages:
        _io_logger.error("[%s] !!! %s / %s | latency_ms=%.0f | %s",
                         request_id, config.provider, config.model, ms, exc)


# --- Plain chat -----------------------------------------------------------


def _chat_raw(config: LLMConfig, system: str, user: str
              ) -> tuple[str, dict[str, Any]]:
    """The reply text and its token usage."""
    if config.provider == "ollama":
        response = _ollama().chat(
            model=config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=_ollama_options(),
        )
        return response["message"]["content"], _usage("ollama", response)

    if config.provider == "gemini":
        # Keep a reference: an unreferenced Client is garbage-collected
        # and closes its connection mid-request.
        client = _gemini_client(config)
        response = client.models.generate_content(
            model=config.model,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system, thinking_config=_gemini_thinking()
            ),
        )
        usage = _usage("gemini", response)
        if not response.candidates:
            return "The model declined to answer this request.", usage
        return _gemini_text(response.candidates[0].content), usage

    if config.provider == "claude":
        response = _claude_request(
            config,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _claude_text(response), _usage("claude", response)

    client = openai.OpenAI(api_key=config.api_key)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "",
            _usage("openai", response))


def _check_key(config: LLMConfig) -> None:
    if config.provider != "ollama" and not config.api_key:
        raise LLMError(f"An API key is required for {config.provider}.")


def check_connection(config: LLMConfig) -> str:
    """Verify the key and model name before any question is asked.

    Uses each provider's "look up model" call, which is free: it proves
    the key works and the model exists without generating any tokens.
    Returns a status line; raises LLMError with the reason on failure.
    """
    _check_key(config)
    name = _PROVIDER_NAMES[config.provider]
    with _provider_errors(config):
        if config.provider == "claude":
            client = anthropic.Anthropic(api_key=config.api_key)
            client.models.retrieve(config.model)
        elif config.provider == "openai":
            client = openai.OpenAI(api_key=config.api_key)
            client.models.retrieve(config.model)
        elif config.provider == "gemini":
            client = _gemini_client(config)
            client.models.get(model=config.model)
        elif config.model not in ollama_models():
            raise LLMError(f"Ollama has no local model named {config.model}")
    logger.info("Connection ok | %s", config)
    return f"Connected to {name} ({config.model})"


def chat(config: LLMConfig, system: str, user: str) -> str:
    """Send one system + user message to the configured provider."""
    _check_key(config)
    _log_request(config, "chat", [{"role": "system", "content": system},
                                  {"role": "user", "content": user}])
    start = time.perf_counter()
    try:
        with _provider_errors(config):
            text, usage = _chat_raw(config, system, user)
    except LLMError as exc:
        _log_failure(config, "LLM call", start, exc)
        raise
    _log_reply(config, "LLM call ok", start, usage, text)
    return text


# --- Tool calling -----------------------------------------------------------
# Tools are defined once in a neutral shape:
#   {"name": ..., "description": ..., "parameters": <JSON schema>}
# That is already OpenAI's "function" object; Claude calls the schema
# field "input_schema".


def _recover_text_tool_calls(
    text: str, tool_names: set[str]
) -> list[ToolCall]:
    """Find tool calls a model printed as JSON text, e.g.
    {"name": "get_machine_health", "arguments": {"asset_id": "P-07"}}.
    Small local models do this instead of emitting a structured call;
    without recovery the agent would show the JSON as its answer."""
    decoder = json.JSONDecoder()
    calls: list[ToolCall] = []
    pos = 0
    while (start := text.find("{", pos)) != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if (
            isinstance(obj, dict)
            and obj.get("name") in tool_names
            and isinstance(obj.get("arguments"), dict)
        ):
            calls.append(
                ToolCall(f"call_{len(calls)}", obj["name"], obj["arguments"])
            )
        pos = end
    return calls


def _tools_ollama(config, system, messages, tools) -> ToolTurn:
    response = _ollama().chat(
        model=config.model,
        messages=[{"role": "system", "content": system}, *messages],
        tools=[{"type": "function", "function": t} for t in tools],
        # Low temperature: the same request takes the same tool path
        # every time, which small local models otherwise don't.
        options=_ollama_options(),
    )
    msg = response.message
    text = msg.content or ""
    calls = [
        ToolCall(f"call_{i}", c.function.name, dict(c.function.arguments))
        for i, c in enumerate(msg.tool_calls or [])
    ]
    if not calls:
        calls = _recover_text_tool_calls(text, {t["name"] for t in tools})
        if calls:
            logger.warning("Recovered %d tool call(s) written as text",
                           len(calls))
            text = ""

    assistant = {"role": "assistant", "content": text}
    if calls:
        assistant["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ]
    return ToolTurn(text, calls, assistant, _usage("ollama", response))


def _tools_openai(config, system, messages, tools) -> ToolTurn:
    client = openai.OpenAI(api_key=config.api_key)
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "system", "content": system}, *messages],
        tools=[{"type": "function", "function": t} for t in tools],
    )
    msg = response.choices[0].message
    raw_calls = msg.tool_calls or []
    calls = [
        ToolCall(c.id, c.function.name, json.loads(c.function.arguments))
        for c in raw_calls
    ]
    assistant = {"role": "assistant", "content": msg.content}
    if raw_calls:
        assistant["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments,
                },
            }
            for c in raw_calls
        ]
    return ToolTurn(msg.content or "", calls, assistant,
                    _usage("openai", response))


def _tools_claude(config, system, messages, tools) -> ToolTurn:
    response = _claude_request(
        config,
        system=system,
        messages=messages,
        tools=[
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ],
    )
    calls = [
        ToolCall(b.id, b.name, dict(b.input))
        for b in response.content
        if b.type == "tool_use"
    ]
    # Send the full content back (thinking + tool_use blocks included).
    assistant = {"role": "assistant", "content": response.content}
    return ToolTurn(_claude_text(response), calls, assistant,
                    _usage("claude", response))


def _tools_gemini(config, system, messages, tools) -> ToolTurn:
    # The agent starts history with a plain {"role", "content"} dict;
    # later turns are already Gemini Content objects.
    contents = [
        genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=m["content"])]
        )
        if isinstance(m, dict) else m
        for m in messages
    ]
    client = _gemini_client(config)
    response = client.models.generate_content(
        model=config.model,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=_gemini_thinking(),
            tools=[genai_types.Tool(function_declarations=[
                genai_types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters_json_schema=t["parameters"],
                )
                for t in tools
            ])],
            # Our code runs the tools (and the approval gate), not the SDK.
            automatic_function_calling=(
                genai_types.AutomaticFunctionCallingConfig(disable=True)
            ),
        ),
    )
    usage = _usage("gemini", response)
    if not response.candidates:
        text = "The model declined to answer this request."
        return ToolTurn(text, [], {"role": "model"}, usage)
    calls = [
        ToolCall(fc.id or "", fc.name, dict(fc.args or {}))
        for fc in (response.function_calls or [])
    ]
    # Send the model's turn back unchanged: it carries Gemini's thought
    # signatures, which multi-step tool use depends on.
    content = response.candidates[0].content
    return ToolTurn(_gemini_text(content), calls, content, usage)


_TOOL_PROVIDERS = {
    "ollama": _tools_ollama,
    "openai": _tools_openai,
    "claude": _tools_claude,
    "gemini": _tools_gemini,
}


def chat_with_tools(
    config: LLMConfig,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> ToolTurn:
    """Run one agent step. `messages` holds provider-native history."""
    _check_key(config)
    _log_request(config, "agent step",
                 [{"role": "system", "content": system}, *messages], tools)
    start = time.perf_counter()
    try:
        with _provider_errors(config):
            turn = _TOOL_PROVIDERS[config.provider](
                config, system, messages, tools
            )
    except LLMError as exc:
        _log_failure(config, "LLM tool step", start, exc)
        raise
    _log_reply(config, "LLM tool step ok", start, turn.usage, turn.text,
               turn.tool_calls)
    return turn


def tool_result_messages(
    config: LLMConfig, results: list[tuple[ToolCall, str]]
) -> list[dict[str, Any]]:
    """Wrap tool outputs in the provider's expected message format."""
    if config.provider == "claude":
        # All results for one step go back in a single user message.
        return [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": c.id, "content": r}
                for c, r in results
            ],
        }]
    if config.provider == "openai":
        return [
            {"role": "tool", "tool_call_id": c.id, "content": r}
            for c, r in results
        ]
    if config.provider == "gemini":
        return [genai_types.Content(role="user", parts=[
            genai_types.Part(function_response=genai_types.FunctionResponse(
                id=c.id or None, name=c.name, response={"result": r}
            ))
            for c, r in results
        ])]
    return [
        {"role": "tool", "tool_name": c.name, "content": r}
        for c, r in results
    ]
