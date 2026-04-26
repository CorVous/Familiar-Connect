"""Tests for :meth:`familiar_connect.llm.LLMClient.chat_stream`.

Barge-in requires token-level streaming so TTS can begin on the first
delta and cancellation actually saves work.
"""

from __future__ import annotations

import asyncio
from typing import Self
from unittest.mock import MagicMock

import httpx
import pytest

import familiar_connect.llm as llm_module
from familiar_connect.llm import LLMClient, Message


def _sse_lines(deltas: list[str]) -> list[bytes]:
    """Build a minimal OpenRouter/OpenAI SSE byte stream."""
    lines: list[bytes] = []
    for delta in deltas:
        payload = (
            '{"choices": [{"delta": {"role": "assistant", "content": "'
            + delta
            + '"}}]}'
        )
        lines.append(b"data: " + payload.encode() + b"\n\n")
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


class TestChatStream:
    @pytest.mark.asyncio
    async def test_yields_each_delta(self) -> None:
        client = LLMClient(api_key="k", model="m")
        chunks = _sse_lines(["Hel", "lo", ", world"])
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
        client._http = fake_http  # type: ignore[assignment]

        got: list[str] = [
            delta
            async for delta in client.chat_stream([Message(role="user", content="hi")])
        ]

        assert got == ["Hel", "lo", ", world"]

    @pytest.mark.asyncio
    async def test_cancel_mid_stream_stops_iteration(self) -> None:
        client = LLMClient(api_key="k", model="m")
        # infinite-ish stream: 100 deltas, cancel after 2
        chunks = _sse_lines([f"d{i}" for i in range(100)])
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
        client._http = fake_http  # type: ignore[assignment]

        got: list[str] = []

        async def reader() -> None:
            async for delta in client.chat_stream([Message(role="user", content="hi")]):
                got.append(delta)
                if len(got) >= 2:
                    return

        await asyncio.wait_for(reader(), timeout=1.0)
        assert got == ["d0", "d1"]

    @pytest.mark.asyncio
    async def test_sends_stream_true_in_payload(self) -> None:
        client = LLMClient(api_key="k", model="m")
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(_sse_lines([])))
        client._http = fake_http  # type: ignore[assignment]

        async for _ in client.chat_stream([Message(role="user", content="x")]):
            pass

        call = fake_http.stream.call_args
        assert call.args[0] == "POST"
        assert call.kwargs["json"]["stream"] is True

    @pytest.mark.asyncio
    async def test_semaphore_released_on_cancel(self) -> None:
        """A cancelled stream must release the rate-limit slot promptly."""
        # Reset the module-level semaphore for test isolation.
        llm_module._request_semaphore = None
        sem = llm_module.get_request_semaphore(max_concurrent=1)
        assert sem._value == 1

        client = LLMClient(api_key="k", model="m")
        chunks = _sse_lines([f"d{i}" for i in range(100)])
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
        client._http = fake_http  # type: ignore[assignment]

        async def reader() -> None:
            async for _delta in client.chat_stream([
                Message(role="user", content="hi")
            ]):
                return  # stop after first delta

        await reader()
        # After the reader exited, slot must be released
        assert sem._value == 1

        llm_module._request_semaphore = None  # cleanup


class TestChatStreamNoHttp:
    """Stubs for code paths that don't involve the HTTP client."""

    @pytest.mark.asyncio
    async def test_chat_stream_fails_on_429_without_retry_loop(
        self,
    ) -> None:
        """A 429 on stream is surfaced as :class:`httpx.HTTPStatusError`.

        ``chat_stream`` intentionally does not inherit the retry loop
        from ``chat`` — retries for a streaming call would hold the
        rate-limit slot during sleeps and starve barge-in cancellation.
        """
        client = LLMClient(api_key="k", model="m")

        fake_http = MagicMock()
        fake_http.stream = MagicMock(
            return_value=_FakeStreamResponse([], status_code=429)
        )
        client._http = fake_http  # type: ignore[assignment]

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in client.chat_stream([Message(role="user", content="x")]):
                pass


class TestSSEErrorFrames:
    """OpenRouter / OpenAI-compatible servers can send mid-stream errors.

    Format observed in the wild::

        data: {"error": {"message": "model not found", "code": 404}}

    Our parser used to silently drop these; we now log a warning so
    the operator sees *why* the reply came back empty.
    """

    @pytest.mark.asyncio
    async def test_top_level_error_frame_logged(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = LLMClient(api_key="k", model="m")
        chunks = [
            b'data: {"error": {"message": "model not found", "code": 404}}\n\n',
            b"data: [DONE]\n\n",
        ]
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
        client._http = fake_http  # type: ignore[assignment]

        with caplog.at_level("WARNING", logger="familiar_connect.llm"):
            got = [
                d async for d in client.chat_stream([Message(role="user", content="x")])
            ]

        assert got == []
        assert any("model not found" in r.getMessage() for r in caplog.records), [
            r.getMessage() for r in caplog.records
        ]
