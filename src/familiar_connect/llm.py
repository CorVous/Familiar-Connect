"""LLM client for generating AI responses via OpenRouter."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from familiar_connect import log_style as ls
from familiar_connect.config import LLM_SLOT_NAMES
from familiar_connect.diagnostics.collector import get_span_collector

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any, Self

    from familiar_connect.config import CharacterConfig

_logger = logging.getLogger(__name__)

_NAME_ALLOWED = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_name(name: str) -> str | None:
    """Sanitize *name* for OpenAI name field, or ``None`` if empty after cleanup.

    Requires ``^[a-zA-Z0-9_-]{1,64}$``; unsupported chars replaced with underscores.
    """
    sanitized = _NAME_ALLOWED.sub("_", name)[:64].strip("_")
    return sanitized or None


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Retry / concurrency constants ---

_MAX_RETRIES = 4
_BASE_DELAY_S = 1.0
_MAX_DELAY_S = 30.0
_DEFAULT_MAX_CONCURRENT = 4

_request_semaphore: asyncio.Semaphore | None = None


def get_request_semaphore(
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
) -> asyncio.Semaphore:
    """Return module-level semaphore, creating on first call.

    Shared across all :class:`LLMClient` instances — bottleneck is
    the OpenRouter API key's rate limit, not any single client.
    """
    global _request_semaphore  # noqa: PLW0603
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(max_concurrent)
    return _request_semaphore


@dataclass
class Message:
    """A single chat message with an optional speaker name."""

    role: str
    content: str
    name: str | None = None

    def to_dict(self: Self) -> dict[str, str]:
        """Serialize to the OpenAI-compatible message dict format."""
        d: dict[str, str] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class SystemPromptLayers:
    """Layers composing the system prompt.

    ``recent_history`` is kept as separate messages, not in system prompt.
    """

    core_instructions: str
    character_card: str = ""
    rag_context: str = ""
    conversation_summary: str = ""
    recent_history: list[Message] = field(default_factory=list)


def build_system_prompt(layers: SystemPromptLayers) -> str:
    """Assemble the system prompt from non-empty layers in priority order."""
    sections: list[str] = []

    if layers.core_instructions:
        sections.append(layers.core_instructions)
    if layers.character_card:
        sections.append(layers.character_card)
    if layers.rag_context:
        sections.append(layers.rag_context)
    if layers.conversation_summary:
        sections.append(layers.conversation_summary)

    return "\n\n".join(sections)


class LLMClient:
    """Client for sending chat completions to OpenRouter."""

    def __init__(
        self: Self,
        *,
        api_key: str,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float | None = None,
        slot: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        # call-site label — appears in span names + the per-call log.
        # ``None`` keeps backward-compat for tests / one-off clients.
        self.slot = slot
        self._http: httpx.AsyncClient | None = None

    def _get_http(self: Self) -> httpx.AsyncClient:
        """Lazily create and return shared HTTP client."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=120.0)
        return self._http

    async def close(self: Self) -> None:
        """Shut down HTTP connection pool."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def build_headers(self: Self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_payload(
        self: Self,
        messages: list[Message],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    async def _post(
        self: Self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        http = self._get_http()
        response: httpx.Response | None = None

        for attempt in range(_MAX_RETRIES + 1):
            async with get_request_semaphore():
                response = await http.post(url, headers=headers, json=payload)

            if response.status_code != 429:
                return response

            # last attempt — don't sleep, just return the 429
            if attempt == _MAX_RETRIES:
                break

            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = min(float(retry_after), _MAX_DELAY_S)
                except ValueError:
                    delay = min(_BASE_DELAY_S * 2**attempt, _MAX_DELAY_S)
            else:
                delay = min(_BASE_DELAY_S * 2**attempt, _MAX_DELAY_S)

            _logger.warning(
                "429 from %s (attempt %d/%d), retrying in %.1fs",
                url,
                attempt + 1,
                _MAX_RETRIES + 1,
                delay,
            )
            await asyncio.sleep(delay)

        # all retries exhausted — caller's raise_for_status() will handle it
        assert response is not None  # noqa: S101 — loop always runs at least once
        return response

    async def chat(self: Self, messages: list[Message]) -> Message:
        """Send messages to OpenRouter and return the assistant's reply."""
        url = f"{self.base_url}/chat/completions"
        headers = self.build_headers()
        payload = self.build_payload(messages)

        response = await self._post(url, headers, payload)
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            msg = "No choices returned from the API"
            raise ValueError(msg)

        reply = choices[0]["message"]
        return Message(role=reply["role"], content=reply["content"])

    async def chat_stream(
        self: Self,
        messages: list[Message],
    ) -> AsyncIterator[str]:
        """Stream assistant content deltas.

        Required for barge-in: TTS can begin on the first delta and
        cancellation mid-stream actually saves work.

        Unlike :meth:`chat`, no retry loop — retries on a streaming
        call would hold the rate-limit slot during sleeps and starve
        cancellation. On 429 the caller sees
        :class:`httpx.HTTPStatusError`.

        Yields:
            str: content delta as it arrives from the server.

        """
        url = f"{self.base_url}/chat/completions"
        headers = self.build_headers()
        payload = self.build_payload(messages)
        payload["stream"] = True
        # Ask OpenRouter to send a final chunk with token usage + the
        # provider that actually served the call. Cost is one extra
        # SSE chunk, value is per-call cache/route visibility.
        payload["usage"] = {"include": True}

        http = self._get_http()
        # Acquire the rate-limit slot only for the HTTP request *initiation*.
        # Holding the semaphore across the entire stream would starve
        # barge-in cancellation: a stalled reader would pin the slot.
        # Release as soon as the server has accepted the request.
        metrics = _CallMetrics(slot=self.slot, model=self.model)
        metrics.input_chars = sum(len(m.content) for m in messages)
        metrics.t_start = time.perf_counter()
        stream_cm = http.stream("POST", url, headers=headers, json=payload)
        semaphore = get_request_semaphore()
        await semaphore.acquire()
        response = None
        try:
            response = await stream_cm.__aenter__()  # noqa: PLC2801
            response.raise_for_status()
        except BaseException:
            if response is not None:
                with contextlib.suppress(Exception):
                    await stream_cm.__aexit__(None, None, None)
            semaphore.release()
            metrics.status = "error"
            metrics.emit()
            raise
        semaphore.release()
        metrics.t_first_byte = time.perf_counter()
        try:
            async for line in response.aiter_lines():
                event = _parse_sse_event(line)
                if event is None:
                    continue
                metrics.absorb(event)
                for delta in _content_deltas(event):
                    if metrics.t_first_delta is None:
                        metrics.t_first_delta = time.perf_counter()
                    yield delta
        except BaseException:
            metrics.status = "error"
            raise
        finally:
            metrics.t_end = time.perf_counter()
            with contextlib.suppress(Exception):
                await stream_cm.__aexit__(None, None, None)
            metrics.emit()


def _parse_sse_event(line: str) -> dict[str, Any] | None:
    """Decode one SSE ``data:`` line into a JSON object.

    Returns ``None`` for blanks, comments, ``[DONE]``, or unparseable
    payloads. Surfaces server-side ``error`` frames as a warning log
    and returns ``None`` so the caller treats them as a non-event.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    err = obj.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or "unknown"
        code = err.get("code")
        _logger.warning(
            f"{ls.tag('LLM', ls.R)} "
            f"{ls.kv('sse_error', str(msg), vc=ls.R)} "
            f"{ls.kv('code', str(code), vc=ls.LW)}"
        )
        return None
    return obj


def _content_deltas(event: dict[str, Any]) -> list[str]:
    """Extract assistant content deltas from a parsed SSE chunk."""
    out: list[str] = []
    for choice in event.get("choices") or []:
        delta = (choice.get("delta") or {}).get("content")
        if isinstance(delta, str) and delta:
            out.append(delta)
    return out


def _parse_sse_deltas(line: str) -> list[str]:
    """Backward-compat helper — parse a line and return content deltas only."""
    event = _parse_sse_event(line)
    return [] if event is None else _content_deltas(event)


@dataclass
class _CallMetrics:
    """Per-call timing + token signals for an OpenRouter request."""

    slot: str | None
    model: str
    input_chars: int = 0
    t_start: float = 0.0
    t_first_byte: float | None = None
    t_first_delta: float | None = None
    t_end: float | None = None
    status: str = "ok"
    provider: str | None = None
    in_tokens: int | None = None
    out_tokens: int | None = None
    cached_tokens: int | None = None

    def absorb(self, event: dict[str, Any]) -> None:
        """Pull provider + usage off any chunk that carries them."""
        provider = event.get("provider")
        if isinstance(provider, str):
            self.provider = provider
        usage = event.get("usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if isinstance(pt, int):
                self.in_tokens = pt
            if isinstance(ct, int):
                self.out_tokens = ct
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cached = details.get("cached_tokens")
                if isinstance(cached, int):
                    self.cached_tokens = cached

    def emit(self) -> None:
        """One structured log line + per-phase spans into the collector."""
        suffix = f".{self.slot}" if self.slot else ""
        ttfb_ms: int | None = None
        ttft_ms: int | None = None
        total_ms: int | None = None
        if self.t_first_byte is not None:
            ttfb_ms = max(0, round((self.t_first_byte - self.t_start) * 1000))
            self._span(f"llm.ttfb{suffix}", ttfb_ms)
        if self.t_first_delta is not None:
            ttft_ms = max(0, round((self.t_first_delta - self.t_start) * 1000))
            self._span(f"llm.ttft{suffix}", ttft_ms)
        if self.t_end is not None:
            total_ms = max(0, round((self.t_end - self.t_start) * 1000))
            self._span(f"llm.total{suffix}", total_ms)

        parts = [
            ls.tag("LLM call", ls.LM),
            ls.kv("slot", self.slot or "-", vc=ls.LC),
            ls.kv("model", self.model, vc=ls.LW),
            ls.kv("status", self.status, vc=ls.LG if self.status == "ok" else ls.R),
            ls.kv("chars", str(self.input_chars), vc=ls.LC),
        ]
        if ttfb_ms is not None:
            parts.append(ls.kv("ttfb_ms", str(ttfb_ms), vc=ls.LC))
        if ttft_ms is not None:
            parts.append(ls.kv("ttft_ms", str(ttft_ms), vc=ls.LC))
        if total_ms is not None:
            parts.append(ls.kv("total_ms", str(total_ms), vc=ls.LC))
        if self.provider is not None:
            parts.append(ls.kv("provider", self.provider, vc=ls.LM))
        if self.in_tokens is not None:
            parts.append(ls.kv("in_tokens", str(self.in_tokens), vc=ls.LW))
        if self.out_tokens is not None:
            parts.append(ls.kv("out_tokens", str(self.out_tokens), vc=ls.LW))
        if self.cached_tokens is not None:
            parts.append(ls.kv("cached", str(self.cached_tokens), vc=ls.LW))
        _logger.info(" ".join(parts))

    @staticmethod
    def _span(name: str, ms: int) -> None:
        with contextlib.suppress(Exception):
            get_span_collector().record(name=name, ms=ms, status="ok")


def create_llm_clients(
    api_key: str,
    character_config: CharacterConfig,
) -> dict[str, LLMClient]:
    """Build one :class:`LLMClient` per call-site slot.

    All clients share one API key and the process-wide rate-limit
    semaphore in :func:`get_request_semaphore`.
    """
    clients: dict[str, LLMClient] = {}
    for slot_name in LLM_SLOT_NAMES:
        slot = character_config.llm[slot_name]
        clients[slot_name] = LLMClient(
            api_key=api_key,
            model=slot.model,
            temperature=slot.temperature,
            slot=slot_name,
        )
        temp = slot.temperature if slot.temperature is not None else "default"
        _logger.info(
            f"{ls.tag('Config', ls.W)} "
            f"{ls.kv('slot', slot_name)} "
            f"{ls.kv('model', str(slot.model))} "
            f"{ls.kv('temperature', str(temp))}"
        )
    return clients
