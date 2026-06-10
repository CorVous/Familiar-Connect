"""LLM client — chat completions via OpenRouter."""

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
    """Sanitize for OpenAI ``name`` field; ``None`` when empty after cleanup.

    Pattern ``^[a-zA-Z0-9_-]{1,64}$``; unsupported chars → underscore.
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
    """Module-level semaphore; lazy-init.

    Shared across all :class:`LLMClient` instances — bottleneck is
    OpenRouter API key's rate limit, not any single client.
    """
    global _request_semaphore  # noqa: PLW0603
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(max_concurrent)
    return _request_semaphore


@dataclass
class Message:
    """Chat message — content + optional speaker name + tool fields.

    ``content`` may be a list of content blocks for multimodal/tool-result
    messages (e.g. vision image_url blocks). str for all other cases.
    """

    role: str
    content: str | list[dict[str, Any]]
    name: str | None = None
    # assistant turns invoking tools — list of OpenAI ``tool_calls`` dicts
    # ``{"id", "type":"function", "function":{"name","arguments"}}``.
    tool_calls: list[dict[str, Any]] | None = None
    # tool-role turns reference the call they answered
    tool_call_id: str | None = None

    @property
    def content_str(self: Self) -> str:
        """Return content as plain text; joins text blocks from multimodal lists."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    def to_dict(self: Self) -> dict[str, Any]:
        """Serialize to OpenAI-compatible message dict."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class LLMDelta:
    """One streaming chunk: content text and/or tool-call fragments."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class SystemPromptLayers:
    """Layers composing system prompt.

    ``recent_history`` stays as separate messages, not in system prompt.
    """

    character_card: str = ""
    rag_context: str = ""
    conversation_summary: str = ""
    recent_history: list[Message] = field(default_factory=list)


def build_system_prompt(layers: SystemPromptLayers) -> str:
    """Assemble from non-empty layers in priority order."""
    sections: list[str] = []

    if layers.character_card:
        sections.append(layers.character_card)
    if layers.rag_context:
        sections.append(layers.rag_context)
    if layers.conversation_summary:
        sections.append(layers.conversation_summary)

    return "\n\n".join(sections)


class LLMClient:
    """OpenRouter chat-completion client."""

    def __init__(
        self: Self,
        *,
        api_key: str,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float | None = None,
        slot: str | None = None,
        provider_order: tuple[str, ...] | None = None,
        provider_allow_fallbacks: bool = True,
        reasoning: str | None = None,
        tool_calling: bool = False,
        image_tools: bool = False,
        multimodal: bool = False,
        no_stream: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        # call-site label — surfaces in span names + per-call log.
        # ``None`` keeps backward-compat for tests / one-off clients.
        self.slot = slot
        # OpenRouter routing override; ``None`` = default. See
        # LLMSlotConfig for rationale + sunset note.
        self.provider_order = provider_order
        self.provider_allow_fallbacks = provider_allow_fallbacks
        # OpenRouter reasoning effort; ``None`` = model default;
        # ``"off"`` → ``exclude=True``; ``"low"|"medium"|"high"`` → ``effort=…``.
        self.reasoning = reasoning
        # gate for agentic-loop / tool-registry plumbing. responders
        # read to decide whether to install tools on a call.
        self.tool_calling_enabled = tool_calling
        # gate for view_image registration (independent of tool_calling)
        self.image_tools_enabled = image_tools
        # whether to send image content blocks in tool-result messages
        self.multimodal = multimodal
        # skip SSE streaming; fixes tool-call-as-text on models with
        # streaming + thinking-off interaction bugs (e.g. Qwen3 vLLM)
        self.no_stream = no_stream
        self._http: httpx.AsyncClient | None = None

    def _get_http(self: Self) -> httpx.AsyncClient:
        """Lazy-init shared HTTP client."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=120.0)
        return self._http

    async def close(self: Self) -> None:
        """Shut down HTTP pool."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def build_headers(self: Self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _log_http_error_body(self: Self, status_code: int, body: object) -> None:
        """Surface upstream error body on 4xx/5xx — not just bare status.

        Without this, ``raise_for_status`` only carries status code +
        reason phrase, hiding payloads like
        ``{"error": {"message": "Unsupported value: 'temperature' …"}}``.
        """
        slot_suffix = f".{self.slot}" if self.slot else ""
        # mock response objects in tests may return non-string ``.text``;
        # coerce so logging never crashes request path.
        body_str = body if isinstance(body, str) else ""
        _logger.warning(
            f"{ls.tag('LLM', ls.R)} "
            f"{ls.kv(f'http_error{slot_suffix}', str(status_code), vc=ls.R)} "
            f"{ls.kv('model', self.model, vc=ls.LW)} "
            f"{ls.kv('body', ls.trunc(body_str, limit=600), vc=ls.LW)}"
        )

    def build_payload(
        self: Self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.provider_order is not None:
            payload["provider"] = {
                "order": list(self.provider_order),
                "allow_fallbacks": self.provider_allow_fallbacks,
            }
        if self.reasoning == "off":
            payload["reasoning"] = {"exclude": True}
        elif self.reasoning is not None:
            payload["reasoning"] = {"effort": self.reasoning}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
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

            # last attempt — don't sleep, just return 429
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

        # all retries exhausted — caller's raise_for_status() handles it
        assert response is not None  # noqa: S101 — loop always runs at least once
        return response

    async def chat(
        self: Self,
        messages: list[Message],
    ) -> Message:
        """Send messages; return assistant reply.

        When model elects to call tools, returned :class:`Message` has
        ``tool_calls`` populated; ``content`` may be empty.
        """
        url = f"{self.base_url}/chat/completions"
        headers = self.build_headers()
        payload = self.build_payload(messages)

        response = await self._post(url, headers, payload)
        if response.status_code >= 400:
            self._log_http_error_body(response.status_code, response.text)
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            msg = "No choices returned from the API"
            raise ValueError(msg)

        reply = choices[0]["message"]
        # ``content`` may be ``None`` when model only emitted tool_calls.
        # normalize to empty string so callers treat content uniformly.
        content = reply.get("content") or ""
        tc = reply.get("tool_calls")
        tc_list: list[dict[str, Any]] | None = None
        if isinstance(tc, list) and tc:
            tc_list = [item for item in tc if isinstance(item, dict)]
        return Message(role=reply["role"], content=content, tool_calls=tc_list)

    async def stream_completion(
        self: Self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMDelta]:
        """Stream assistant deltas.

        Barge-in supported via generator close / cancel — no retry loop
        holds rate-limit slot through sleeps.

        Yields:
            one :class:`LLMDelta` per parsed SSE chunk carrying content
            text, tool-call fragments, or finish reason.

        """
        if self.no_stream:
            msg = await self.chat(messages)
            if isinstance(msg.content, str) and msg.content:
                yield LLMDelta(content=msg.content)
            if msg.tool_calls:
                for i, tc in enumerate(msg.tool_calls):
                    fn = tc.get("function", {})
                    yield LLMDelta(
                        tool_calls=[
                            {
                                "index": i,
                                "id": tc.get("id", f"call_{i}"),
                                "type": "function",
                                "function": {
                                    "name": fn.get("name", ""),
                                    "arguments": fn.get("arguments", "{}"),
                                },
                            }
                        ]
                    )
            yield LLMDelta(finish_reason="stop" if not msg.tool_calls else "tool_calls")
            return

        url = f"{self.base_url}/chat/completions"
        headers = self.build_headers()
        payload = self.build_payload(messages, tools=tools)
        payload["stream"] = True
        # ask OpenRouter for trailing chunk carrying token usage +
        # provider that served call. costs one extra SSE chunk for
        # per-call cache/route visibility.
        payload["usage"] = {"include": True}

        http = self._get_http()
        metrics = _CallMetrics(slot=self.slot, model=self.model)
        metrics.input_chars = sum(
            len(m.content) if isinstance(m.content, str) else 0 for m in messages
        )
        metrics.t_start = time.perf_counter()
        stream_cm = http.stream("POST", url, headers=headers, json=payload)
        semaphore = get_request_semaphore()
        await semaphore.acquire()
        response = None
        try:
            response = await stream_cm.__aenter__()  # noqa: PLC2801
            if response.status_code >= 400:
                with contextlib.suppress(Exception):
                    await response.aread()
                body = getattr(response, "text", "") or ""
                self._log_http_error_body(response.status_code, body)
            response.raise_for_status()
        except (GeneratorExit, asyncio.CancelledError):
            if response is not None:
                with contextlib.suppress(Exception):
                    await stream_cm.__aexit__(None, None, None)
            semaphore.release()
            metrics.status = "cancelled"
            metrics.emit()
            raise
        except Exception:
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
                content_parts = _content_deltas(event)
                tool_call_parts = _tool_call_deltas(event)
                finish = _finish_reason(event)
                if not content_parts and not tool_call_parts and finish is None:
                    continue
                joined_content = "".join(content_parts)
                if joined_content and metrics.t_first_delta is None:
                    metrics.t_first_delta = time.perf_counter()
                yield LLMDelta(
                    content=joined_content,
                    tool_calls=tool_call_parts,
                    finish_reason=finish,
                )
        except (GeneratorExit, asyncio.CancelledError):
            # caller closed generator (barge-in / silent gate / scope
            # cancel). distinct status so /diagnostics shows real
            # provider errors clearly.
            metrics.status = "cancelled"
            raise
        except Exception:
            metrics.status = "error"
            raise
        finally:
            metrics.t_end = time.perf_counter()
            with contextlib.suppress(Exception):
                await stream_cm.__aexit__(None, None, None)
            metrics.emit()

    async def chat_stream(
        self: Self,
        messages: list[Message],
    ) -> AsyncIterator[str]:
        """Stream assistant content deltas as strings.

        Thin wrapper over :meth:`stream_completion`; projects to
        content only. Explicitly closes inner generator on caller
        ``aclose`` so inner ``finally`` (metrics emit, semaphore
        release, body close) actually runs.

        Yields:
            content delta strings as they arrive.

        """
        inner = self.stream_completion(messages)
        try:
            async for delta in inner:
                if delta.content:
                    yield delta.content
        finally:
            # ``stream_completion`` concretely an async generator;
            # ``AsyncIterator`` doesn't promise ``aclose`` but runtime
            # object does. suppress so any non-generator iterator
            # passed via test mock doesn't crash cleanup.
            aclose = getattr(inner, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()


def _parse_sse_event(line: str) -> dict[str, Any] | None:
    """Decode one SSE ``data:`` line into JSON object.

    ``None`` for blanks, comments, ``[DONE]``, or unparseable payloads.
    Surfaces server-side ``error`` frames as warning log; returns
    ``None`` so caller treats them as non-event.
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
    """Extract assistant content deltas from parsed SSE chunk."""
    out: list[str] = []
    for choice in event.get("choices") or []:
        delta = (choice.get("delta") or {}).get("content")
        if isinstance(delta, str) and delta:
            out.append(delta)
    return out


def _tool_call_deltas(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool-call delta fragments from parsed SSE chunk.

    OpenAI/OpenRouter streams emit tool_calls as list with ``index``
    per call; callers must accumulate ``function.arguments`` string
    fragments by index until stream closes.
    """
    out: list[dict[str, Any]] = []
    for choice in event.get("choices") or []:
        tcs = (choice.get("delta") or {}).get("tool_calls")
        if isinstance(tcs, list):
            out.extend(tc for tc in tcs if isinstance(tc, dict))
    return out


def _finish_reason(event: dict[str, Any]) -> str | None:
    """Pull ``finish_reason`` off first choice when present."""
    for choice in event.get("choices") or []:
        fr = choice.get("finish_reason")
        if isinstance(fr, str):
            return fr
    return None


def _parse_sse_deltas(line: str) -> list[str]:
    """Backward-compat helper — parse line, return content deltas only."""
    event = _parse_sse_event(line)
    return [] if event is None else _content_deltas(event)


@dataclass
class _CallMetrics:
    """Per-call timing + token signals for OpenRouter request."""

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
        """Pull provider + usage off any chunk carrying them."""
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
        """One structured log line + per-phase spans into collector."""
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
    """One :class:`LLMClient` per call-site slot.

    All clients share one API key + process-wide rate-limit semaphore
    in :func:`get_request_semaphore`. Reserved key
    ``"__image_description__"`` holds the vision model client when
    ``[llm].image_description_model`` is set.
    """
    clients: dict[str, LLMClient] = {}
    for slot_name in LLM_SLOT_NAMES:
        slot = character_config.llm[slot_name]
        clients[slot_name] = LLMClient(
            api_key=api_key,
            model=slot.model,
            temperature=slot.temperature,
            slot=slot_name,
            provider_order=slot.provider_order,
            provider_allow_fallbacks=slot.provider_allow_fallbacks,
            reasoning=slot.reasoning,
            tool_calling=slot.tool_calling,
            image_tools=slot.image_tools,
            multimodal=slot.multimodal,
        )
        temp = slot.temperature if slot.temperature is not None else "default"
        log_parts = [
            ls.tag("Config", ls.W),
            ls.kv("slot", slot_name),
            ls.kv("model", str(slot.model)),
            ls.kv("temperature", str(temp)),
        ]
        if slot.provider_order is not None:
            log_parts.append(ls.kv("provider_order", ",".join(slot.provider_order)))
            if not slot.provider_allow_fallbacks:
                log_parts.append(ls.kv("fallbacks", "off", vc=ls.LY))
        if slot.reasoning is not None:
            log_parts.append(ls.kv("reasoning", slot.reasoning, vc=ls.LM))
        if slot.tool_calling:
            log_parts.append(ls.kv("tools", "on", vc=ls.LM))
        if slot.image_tools:
            log_parts.append(ls.kv("image_tools", "on", vc=ls.LM))
        if slot.multimodal:
            log_parts.append(ls.kv("multimodal", "on", vc=ls.LM))
        _logger.info(" ".join(log_parts))

    # vision description client — reserved slot, built when configured
    if character_config.image_description_model:
        clients["__image_description__"] = LLMClient(
            api_key=api_key,
            model=character_config.image_description_model,
            slot="image_description",
        )
        _logger.info(
            f"{ls.tag('Config', ls.W)} "
            f"{ls.kv('slot', 'image_description')} "
            f"{ls.kv('model', character_config.image_description_model)}"
        )
    return clients
