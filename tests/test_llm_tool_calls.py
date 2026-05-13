"""Tests for tool-calling on the LLM client.

Covers:
* :class:`Message` extension (``tool_calls``, ``tool_call_id``)
* :class:`LLMDelta` streaming delta dataclass
* ``build_payload`` accepting a ``tools=`` list
* ``chat()`` parsing ``tool_calls`` off the assistant message
* ``stream_completion()`` accumulating tool-call fragments across SSE chunks
"""

from __future__ import annotations

import json
from typing import Self
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

import familiar_connect.llm as llm_module
from familiar_connect.llm import LLMClient, LLMDelta, Message

# ---------------------------------------------------------------------------
# Message extensions
# ---------------------------------------------------------------------------


class TestMessageToolFields:
    def test_assistant_message_carries_tool_calls(self) -> None:
        tc = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "set_alarm", "arguments": '{"reason":"x"}'},
            }
        ]
        msg = Message(role="assistant", content="", tool_calls=tc)
        d = msg.to_dict()
        assert d["tool_calls"] == tc
        # content "" still allowed; OpenRouter accepts empty string
        assert d["role"] == "assistant"

    def test_tool_role_message_carries_tool_call_id(self) -> None:
        msg = Message(role="tool", content='{"ok": true}', tool_call_id="call_1")
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_1"
        assert d["content"] == '{"ok": true}'

    def test_to_dict_omits_tool_fields_when_none(self) -> None:
        msg = Message(role="assistant", content="hi")
        d = msg.to_dict()
        assert "tool_calls" not in d
        assert "tool_call_id" not in d


# ---------------------------------------------------------------------------
# build_payload with tools
# ---------------------------------------------------------------------------


class TestBuildPayloadTools:
    def test_payload_includes_tools_when_passed(self) -> None:
        client = LLMClient(api_key="k", model="m")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_alarm",
                    "description": "Schedule a wake.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        payload = client.build_payload(
            [Message(role="user", content="hi")], tools=tools
        )
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "auto"

    def test_payload_omits_tools_when_empty(self) -> None:
        client = LLMClient(api_key="k", model="m")
        payload = client.build_payload([Message(role="user", content="hi")], tools=[])
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_payload_omits_tools_when_none(self) -> None:
        client = LLMClient(api_key="k", model="m")
        payload = client.build_payload([Message(role="user", content="hi")])
        assert "tools" not in payload


# ---------------------------------------------------------------------------
# chat() parses tool_calls
# ---------------------------------------------------------------------------


class TestChatParsesToolCalls:
    @pytest.mark.asyncio
    async def test_chat_returns_message_with_tool_calls(self) -> None:
        client = LLMClient(api_key="k", model="m")
        tc = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "set_alarm",
                    "arguments": '{"reason":"wakeup","delay_seconds":30}',
                },
            }
        ]
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tc,
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        with patch.object(client, "_post", return_value=mock_resp):
            result = await client.chat([Message(role="user", content="set an alarm")])

        assert result.role == "assistant"
        assert not result.content
        assert result.tool_calls == tc

    @pytest.mark.asyncio
    async def test_chat_without_tool_calls_returns_plain_message(self) -> None:
        client = LLMClient(api_key="k", model="m")
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ]
        }
        with patch.object(client, "_post", return_value=mock_resp):
            result = await client.chat([Message(role="user", content="hi")])

        assert result.content == "hi"
        assert result.tool_calls is None


# ---------------------------------------------------------------------------
# stream_completion accumulates tool-call deltas
# ---------------------------------------------------------------------------


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


class _FakeStreamResponse:
    def __init__(
        self, chunks: list[bytes], status_code: int = 200, body: str = ""
    ) -> None:
        self._chunks = chunks
        self.status_code = status_code
        self._body = body
        self.text = body

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aread(self) -> bytes:
        return self._body.encode()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "bad", request=MagicMock(), response=MagicMock()
            )

    async def aiter_lines(self):
        for chunk in self._chunks:
            yield chunk.decode().rstrip("\n")


@pytest.fixture(autouse=True)
def _reset_semaphore() -> None:
    llm_module._request_semaphore = None


class TestStreamCompletion:
    @pytest.mark.asyncio
    async def test_streams_content_deltas(self) -> None:
        client = LLMClient(api_key="k", model="m")
        chunks = [
            _sse({"choices": [{"delta": {"content": "Hel"}}]}),
            _sse({"choices": [{"delta": {"content": "lo"}}]}),
            b"data: [DONE]\n\n",
        ]
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
        client._http = fake_http  # type: ignore[assignment]

        deltas: list[LLMDelta] = [
            d
            async for d in client.stream_completion([
                Message(role="user", content="hi")
            ])
        ]
        contents = [d.content for d in deltas if d.content]
        assert contents == ["Hel", "lo"]
        # No tool_calls in any delta
        assert all(not d.tool_calls for d in deltas)

    @pytest.mark.asyncio
    async def test_accumulates_tool_call_fragments_across_chunks(self) -> None:
        client = LLMClient(api_key="k", model="m")
        # First chunk: id + name + first arg fragment
        # Second chunk: more arg fragments
        # Third chunk: tail arg fragment
        chunks = [
            _sse({
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_xyz",
                                    "type": "function",
                                    "function": {
                                        "name": "set_alarm",
                                        "arguments": '{"reas',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }),
            _sse({
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'on":"wake","de'},
                                }
                            ]
                        }
                    }
                ]
            }),
            _sse({
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'lay_seconds":30}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }),
            b"data: [DONE]\n\n",
        ]
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
        client._http = fake_http  # type: ignore[assignment]

        # Collect all tool_calls fragments
        seen_tcs: list[list[dict]] = [
            d.tool_calls
            async for d in client.stream_completion([
                Message(role="user", content="set an alarm")
            ])
            if d.tool_calls
        ]

        # At least one delta yielded tool_calls
        assert seen_tcs, "expected at least one tool_call delta"
        # Caller-side accumulation: merge all fragments by index
        merged: dict[int, dict] = {}
        for frame in seen_tcs:
            for tc in frame:
                idx = tc["index"]
                bucket = merged.setdefault(idx, {"function": {"arguments": ""}})
                if "id" in tc:
                    bucket["id"] = tc["id"]
                if "type" in tc:
                    bucket["type"] = tc["type"]
                fn = tc.get("function") or {}
                if "name" in fn:
                    bucket["function"]["name"] = fn["name"]
                if "arguments" in fn:
                    bucket["function"]["arguments"] += fn["arguments"]

        assert merged[0]["id"] == "call_xyz"
        assert merged[0]["function"]["name"] == "set_alarm"
        # Decoded arguments JSON
        args = json.loads(merged[0]["function"]["arguments"])
        assert args == {"reason": "wake", "delay_seconds": 30}

    @pytest.mark.asyncio
    async def test_sends_tools_in_payload(self) -> None:
        client = LLMClient(api_key="k", model="m")
        chunks = [b"data: [DONE]\n\n"]
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
        client._http = fake_http  # type: ignore[assignment]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_alarm",
                    "description": "",
                    "parameters": {"type": "object"},
                },
            }
        ]
        async for _ in client.stream_completion(
            [Message(role="user", content="x")], tools=tools
        ):
            pass

        sent_payload = fake_http.stream.call_args.kwargs["json"]
        assert sent_payload["tools"] == tools
        assert sent_payload["stream"] is True

    @pytest.mark.asyncio
    async def test_yields_finish_reason_on_terminal_chunk(self) -> None:
        client = LLMClient(api_key="k", model="m")
        chunks = [
            _sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
            b"data: [DONE]\n\n",
        ]
        fake_http = MagicMock()
        fake_http.stream = MagicMock(return_value=_FakeStreamResponse(chunks))
        client._http = fake_http  # type: ignore[assignment]

        finish_reasons: list[str | None] = [
            d.finish_reason
            async for d in client.stream_completion([Message(role="user", content="x")])
        ]

        assert "stop" in finish_reasons


# ---------------------------------------------------------------------------
# LLMClient.tool_calling_enabled flag
# ---------------------------------------------------------------------------


class TestToolCallingFlag:
    def test_defaults_off(self) -> None:
        client = LLMClient(api_key="k", model="m")
        assert client.tool_calling_enabled is False

    def test_constructor_enables(self) -> None:
        client = LLMClient(api_key="k", model="m", tool_calling=True)
        assert client.tool_calling_enabled is True
