"""Tests for LLM call diagnostics — TTFB / TTFT / total spans + structured log.

OpenRouter shows huge variance in TTFT depending on model, provider
routing, and prompt cache state. These signals tell prompt-bloat
from routing-tax apart at a glance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Self
from unittest.mock import MagicMock

import httpx
import pytest

from familiar_connect.diagnostics.collector import (
    get_span_collector,
    reset_span_collector,
)
from familiar_connect.llm import LLMClient, Message

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """``ls.kv`` interleaves ANSI escapes between key and value; strip for matching."""
    return _ANSI.sub("", text)


@pytest.fixture(autouse=True)
def _isolate_collector() -> None:
    reset_span_collector()


def _sse_lines(
    deltas: list[str],
    *,
    usage: dict | None = None,
    provider: str | None = None,
) -> list[bytes]:
    """Build a streaming response, optionally with a final usage chunk."""
    lines: list[bytes] = []
    for delta in deltas:
        payload = (
            '{"choices": [{"delta": {"role": "assistant", "content": "'
            + delta
            + '"}}]}'
        )
        lines.append(b"data: " + payload.encode() + b"\n\n")
    if usage is not None or provider is not None:
        chunk: dict[str, object] = {"choices": []}
        if usage is not None:
            chunk["usage"] = usage
        if provider is not None:
            chunk["provider"] = provider
        lines.append(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    lines.append(b"data: [DONE]\n\n")
    return lines


class _FakeStreamResponse:
    """Async-context-manager stand-in for ``httpx.AsyncClient.stream``."""

    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        self._chunks = chunks
        self.status_code = status_code

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "bad", request=MagicMock(), response=MagicMock()
            )

    async def aiter_lines(self):
        for chunk in self._chunks:
            yield chunk.decode().rstrip("\n")


def _wire(
    client: LLMClient,
    chunks: list[bytes],
) -> MagicMock:
    fake_http = MagicMock()
    fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
    client._http = fake_http
    return fake_http


class TestPayloadFlags:
    def test_chat_stream_requests_usage_include(self) -> None:
        """OpenRouter sends a final usage chunk only if asked."""
        client = LLMClient(api_key="k", model="m")
        fake_http = _wire(client, _sse_lines([]))

        async def drain() -> None:
            async for _ in client.chat_stream([Message(role="user", content="hi")]):
                pass

        asyncio.run(drain())

        payload = fake_http.stream.call_args.kwargs["json"]
        assert payload["stream"] is True
        # Permissive check: usage flag is present and truthy.
        assert payload.get("usage") in ({"include": True}, True)


class TestSpansEmitted:
    @pytest.mark.asyncio
    async def test_chat_stream_emits_ttfb_ttft_total_spans(self) -> None:
        client = LLMClient(api_key="k", model="m", slot="main_prose")
        _wire(client, _sse_lines(["Hel", "lo"]))

        async for _ in client.chat_stream([Message(role="user", content="hi")]):
            pass

        names = {r.name for r in get_span_collector().all()}
        assert "llm.ttfb.main_prose" in names
        assert "llm.ttft.main_prose" in names
        assert "llm.total.main_prose" in names

    @pytest.mark.asyncio
    async def test_no_slot_falls_back_to_unsuffixed_span_names(self) -> None:
        """Backward-compat: clients constructed without a slot still produce spans."""
        client = LLMClient(api_key="k", model="m")
        _wire(client, _sse_lines(["x"]))

        async for _ in client.chat_stream([Message(role="user", content="hi")]):
            pass

        names = {r.name for r in get_span_collector().all()}
        assert {"llm.ttfb", "llm.ttft", "llm.total"}.issubset(names)

    @pytest.mark.asyncio
    async def test_no_ttft_when_stream_yields_no_content(self) -> None:
        """Empty SSE stream → ttfb + total recorded; ttft skipped."""
        client = LLMClient(api_key="k", model="m", slot="main_prose")
        _wire(client, _sse_lines([]))

        async for _ in client.chat_stream([Message(role="user", content="hi")]):
            pass

        names = {r.name for r in get_span_collector().all()}
        assert "llm.ttfb.main_prose" in names
        assert "llm.total.main_prose" in names
        assert "llm.ttft.main_prose" not in names


class TestStructuredCallLog:
    @pytest.mark.asyncio
    async def test_call_log_includes_payload_chars_and_metadata(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One structured ``[LLM call]`` line per stream with the key signals."""
        client = LLMClient(api_key="k", model="m", slot="main_prose")
        usage = {"prompt_tokens": 1234, "completion_tokens": 12, "total_tokens": 1246}
        _wire(client, _sse_lines(["Hi"], usage=usage, provider="openai"))

        with caplog.at_level(logging.INFO, logger="familiar_connect.llm"):
            async for _ in client.chat_stream([
                Message(role="system", content="A" * 100),
                Message(role="user", content="hello"),
            ]):
                pass

        msgs = [_strip_ansi(r.getMessage()) for r in caplog.records]
        call_lines = [m for m in msgs if "LLM call" in m]
        assert call_lines
        line = call_lines[0]
        # Char count of input payload: 100 (system) + 5 (user) = 105.
        assert "chars=105" in line
        assert "model=m" in line
        assert "slot=main_prose" in line
        # Final usage chunk surfaced.
        assert "in_tokens=1234" in line
        assert "out_tokens=12" in line
        assert "provider=openai" in line


class TestUsageChunkParsing:
    @pytest.mark.asyncio
    async def test_cached_tokens_surface_when_provider_supports_cache(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Anthropic/OpenAI return ``cached_tokens`` inside ``usage`` for cache hits."""
        client = LLMClient(api_key="k", model="m", slot="main_prose")
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 800},
        }
        _wire(client, _sse_lines(["x"], usage=usage))

        with caplog.at_level(logging.INFO, logger="familiar_connect.llm"):
            async for _ in client.chat_stream([Message(role="user", content="x")]):
                pass

        line = next(
            _strip_ansi(r.getMessage())
            for r in caplog.records
            if "LLM call" in r.getMessage()
        )
        assert "cached=800" in line
