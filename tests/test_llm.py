"""Tests for the LLM client (OpenRouter integration)."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

import familiar_connect.llm as llm_module
from familiar_connect.config import LLM_SLOT_NAMES, load_character_config
from familiar_connect.llm import (
    _MAX_DELAY_S,
    _MAX_RETRIES,
    LLMClient,
    Message,
    SystemPromptLayers,
    build_system_prompt,
    create_llm_clients,
)

if TYPE_CHECKING:
    from pathlib import Path

# --- Message dataclass ---


class TestMessage:
    def test_user_message_has_name(self) -> None:
        """User messages carry the speaker's name."""
        msg = Message(role="user", content="hello", name="Alice")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.name == "Alice"

    def test_assistant_message_has_no_name(self) -> None:
        """Assistant messages don't need a name field."""
        msg = Message(role="assistant", content="hi back")
        assert msg.role == "assistant"
        assert msg.name is None

    def test_system_message(self) -> None:
        """System messages carry instructions."""
        msg = Message(role="system", content="You are a helpful familiar.")
        assert msg.role == "system"

    def test_to_dict_includes_name_when_present(self) -> None:
        """Serialization includes the name field for user messages."""
        msg = Message(role="user", content="hello", name="Bob")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello", "name": "Bob"}

    def test_to_dict_excludes_name_when_none(self) -> None:
        """Serialization omits name when it's None."""
        msg = Message(role="assistant", content="hi")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "hi"}
        assert "name" not in d


# --- System prompt layering ---


class TestSystemPromptLayers:
    def test_layers_have_all_fields(self) -> None:
        """SystemPromptLayers holds all five prompt layers from the plan."""
        layers = SystemPromptLayers(
            core_instructions="Be helpful.",
            character_card="You are Merlin the wizard.",
            rag_context="User Alice likes cats.",
            conversation_summary="They discussed pets.",
            recent_history=[],
        )
        assert layers.core_instructions == "Be helpful."
        assert layers.character_card == "You are Merlin the wizard."
        assert layers.rag_context == "User Alice likes cats."
        assert layers.conversation_summary == "They discussed pets."
        assert layers.recent_history == []

    def test_build_system_prompt_ordering(self) -> None:
        """System prompt sections appear in the correct order."""
        layers = SystemPromptLayers(
            core_instructions="CORE",
            character_card="CHARACTER",
            rag_context="RAG",
            conversation_summary="SUMMARY",
            recent_history=[],
        )
        prompt = build_system_prompt(layers)
        core_pos = prompt.index("CORE")
        char_pos = prompt.index("CHARACTER")
        rag_pos = prompt.index("RAG")
        summary_pos = prompt.index("SUMMARY")
        assert core_pos < char_pos < rag_pos < summary_pos

    def test_build_system_prompt_omits_empty_sections(self) -> None:
        """Empty optional layers are excluded from the system prompt."""
        layers = SystemPromptLayers(
            core_instructions="CORE",
            character_card="",
            rag_context="",
            conversation_summary="",
            recent_history=[],
        )
        prompt = build_system_prompt(layers)
        assert "CORE" in prompt
        # Empty sections should not leave stray headers or blank blocks
        assert "CHARACTER" not in prompt.upper() or prompt.count("\n\n\n") == 0


# --- Multi-user conversation formatting ---


class TestMultiUserMessages:
    def test_multiple_users_before_assistant(self) -> None:
        """Multiple user messages from different speakers precede an assistant turn."""
        messages = [
            Message(role="user", content="What do you think?", name="Alice"),
            Message(role="user", content="I agree with Alice.", name="Bob"),
            Message(role="user", content="Same here.", name="Charlie"),
        ]
        dicts = [m.to_dict() for m in messages]
        assert len(dicts) == 3
        assert all(d["role"] == "user" for d in dicts)
        names = [d["name"] for d in dicts]
        assert names == ["Alice", "Bob", "Charlie"]

    def test_conversation_with_interleaved_assistant(self) -> None:
        """A realistic conversation: multiple users speak, then assistant replies."""
        messages = [
            Message(role="user", content="Hey familiar!", name="Alice"),
            Message(role="user", content="Yeah wake up!", name="Bob"),
            Message(role="assistant", content="Good morning, friends!"),
            Message(role="user", content="What's the weather?", name="Alice"),
            Message(role="assistant", content="I'm not sure, I'm a magical familiar."),
        ]
        dicts = [m.to_dict() for m in messages]
        # First two are users with names, third is assistant without name
        assert dicts[0]["name"] == "Alice"
        assert dicts[1]["name"] == "Bob"
        assert "name" not in dicts[2]


# --- LLMClient ---


class TestLLMClient:
    def test_init_stores_api_key(self) -> None:
        """Client stores the API key it was constructed with."""
        client = LLMClient(api_key="test-key", model="openai/gpt-4o")
        assert client.api_key == "test-key"

    def test_init_custom_model(self) -> None:
        """Client accepts a custom model override."""
        client = LLMClient(api_key="test-key", model="anthropic/claude-sonnet-4")
        assert client.model == "anthropic/claude-sonnet-4"

    def test_init_default_base_url(self) -> None:
        """Client defaults to OpenRouter's API URL."""
        client = LLMClient(api_key="test-key", model="openai/gpt-4o")
        assert "openrouter.ai" in client.base_url

    def test_init_custom_base_url(self) -> None:
        """Client accepts a custom base URL for testing or alternative endpoints."""
        client = LLMClient(
            api_key="test-key",
            model="openai/gpt-4o",
            base_url="https://custom.api/v1",
        )
        assert client.base_url == "https://custom.api/v1"

    def test_builds_request_headers(self) -> None:
        """Request headers include Authorization and required OpenRouter headers."""
        client = LLMClient(api_key="sk-or-test-123", model="openai/gpt-4o")
        headers = client.build_headers()
        assert headers["Authorization"] == "Bearer sk-or-test-123"
        assert "Content-Type" in headers

    def test_builds_request_payload(self) -> None:
        """Request payload has the correct structure for OpenRouter."""
        client = LLMClient(api_key="test-key", model="openai/gpt-4o")
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi", name="Alice"),
        ]
        payload = client.build_payload(messages)
        assert payload["model"] == "openai/gpt-4o"
        assert payload["messages"] == [m.to_dict() for m in messages]

    def test_payload_includes_temperature(self) -> None:
        """Temperature parameter is passed through to the payload."""
        client = LLMClient(
            api_key="test-key",
            model="openai/gpt-4o",
            temperature=0.7,
        )
        messages = [Message(role="user", content="Hi", name="Alice")]
        payload = client.build_payload(messages)
        assert payload["temperature"] == pytest.approx(0.7)


class TestLLMClientChat:
    @pytest.fixture
    def client(self) -> LLMClient:
        return LLMClient(api_key="test-key", model="openai/gpt-4o")

    @pytest.fixture
    def sample_messages(self) -> list[Message]:
        return [
            Message(role="system", content="You are a wizard familiar."),
            Message(role="user", content="Hello familiar!", name="Alice"),
            Message(role="user", content="Yeah hi!", name="Bob"),
        ]

    @pytest.fixture
    def mock_success_response(self) -> dict:
        return {
            "id": "gen-abc123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Greetings, Alice and Bob!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 40,
            },
        }

    def _make_mock_response(
        self,
        json_data: dict,
        *,
        status_code: int = 200,
        raise_error: Exception | None = None,
    ) -> Mock:
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data
        if raise_error:
            mock_response.raise_for_status.side_effect = raise_error
        return mock_response

    @pytest.mark.asyncio
    async def test_chat_returns_assistant_message(
        self,
        client: LLMClient,
        sample_messages: list[Message],
        mock_success_response: dict,
    ) -> None:
        """chat() returns the assistant's response content."""
        mock_response = self._make_mock_response(mock_success_response)

        with patch.object(client, "_post", return_value=mock_response):
            result = await client.chat(sample_messages)

        assert result.content == "Greetings, Alice and Bob!"
        assert result.role == "assistant"

    @pytest.mark.asyncio
    async def test_chat_sends_all_messages(
        self,
        client: LLMClient,
        sample_messages: list[Message],
        mock_success_response: dict,
    ) -> None:
        """chat() sends all messages (system + multi-user) to the API."""
        mock_response = self._make_mock_response(mock_success_response)
        captured_payload: dict = {}

        def fake_post(
            _url: str,
            _headers: dict,
            payload: dict,
        ) -> Mock:
            captured_payload.update(payload)
            return mock_response

        with patch.object(client, "_post", side_effect=fake_post):
            await client.chat(sample_messages)

        sent_messages = captured_payload["messages"]
        assert len(sent_messages) == 3
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1]["name"] == "Alice"
        assert sent_messages[2]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_chat_raises_on_http_error(
        self,
        client: LLMClient,
        sample_messages: list[Message],
    ) -> None:
        """chat() raises on non-200 responses."""
        error = httpx.HTTPStatusError(
            "Rate limited",
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
            response=httpx.Response(429),
        )
        mock_response = self._make_mock_response({}, status_code=429, raise_error=error)

        with (
            patch.object(client, "_post", return_value=mock_response),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await client.chat(sample_messages)

    @pytest.mark.asyncio
    async def test_chat_raises_on_empty_choices(
        self,
        client: LLMClient,
        sample_messages: list[Message],
    ) -> None:
        """chat() raises when the API returns no choices."""
        mock_response = self._make_mock_response({"choices": []})

        with (
            patch.object(client, "_post", return_value=mock_response),
            pytest.raises(ValueError, match=r"[Nn]o.*choices"),
        ):
            await client.chat(sample_messages)


# --- Per-slot factory ---


class TestCreateLLMClients:
    """:func:`create_llm_clients` builds one client per LLM slot."""

    def test_returns_one_client_per_slot(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """Every slot in ``LLM_SLOT_NAMES`` appears in the output dict."""
        cfg = load_character_config(
            tmp_path / "missing.toml",
            defaults_path=default_profile_path,
        )
        clients = create_llm_clients("sk-or-test-abc", cfg)
        assert set(clients.keys()) == set(LLM_SLOT_NAMES)

    def test_each_client_carries_its_slot_model(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """Per-slot ``model`` from the config is passed to each client."""
        cfg = load_character_config(
            tmp_path / "missing.toml",
            defaults_path=default_profile_path,
        )
        clients = create_llm_clients("sk-or-test-abc", cfg)
        for slot_name, client in clients.items():
            assert client.model == cfg.llm[slot_name].model

    def test_each_client_carries_its_slot_temperature(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """Per-slot ``temperature`` from the config is passed to each client."""
        cfg = load_character_config(
            tmp_path / "missing.toml",
            defaults_path=default_profile_path,
        )
        clients = create_llm_clients("sk-or-test-abc", cfg)
        for slot_name, client in clients.items():
            assert client.temperature == cfg.llm[slot_name].temperature

    def test_all_clients_share_api_key(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """Every slot reuses the single ``OPENROUTER_API_KEY`` value."""
        cfg = load_character_config(
            tmp_path / "missing.toml",
            defaults_path=default_profile_path,
        )
        clients = create_llm_clients("sk-or-test-abc", cfg)
        for client in clients.values():
            assert client.api_key == "sk-or-test-abc"

    def test_user_override_changes_slot(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """User config overrides the main_prose slot from the default profile."""
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.main_prose]\nmodel = "user/custom-prose"\ntemperature = 0.9\n',
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        clients = create_llm_clients("sk-or-test-abc", cfg)
        assert clients["main_prose"].model == "user/custom-prose"
        assert clients["main_prose"].temperature == pytest.approx(0.9)

    def test_returns_distinct_client_instances(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """Each slot gets its own :class:`LLMClient`, not a shared instance."""
        cfg = load_character_config(
            tmp_path / "missing.toml",
            defaults_path=default_profile_path,
        )
        clients = create_llm_clients("sk-or-test-abc", cfg)
        instances = list(clients.values())
        assert len({id(c) for c in instances}) == len(instances)


# --- Integration-style test with full prompt assembly ---


class TestPromptAssembly:
    def test_full_prompt_assembly_with_multi_user(self) -> None:
        """Assemble a complete message list: system prompt + multi-user history."""
        layers = SystemPromptLayers(
            core_instructions="You are a helpful AI familiar.",
            character_card="Personality: cheerful wizard's companion.",
            rag_context="Alice's favorite color is blue.",
            conversation_summary="The group was discussing hobbies.",
            recent_history=[],
        )
        system_prompt = build_system_prompt(layers)

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content="I love painting!", name="Alice"),
            Message(role="user", content="I prefer music.", name="Bob"),
            Message(role="user", content="What about you, familiar?", name="Charlie"),
        ]

        dicts = [m.to_dict() for m in messages]

        # System message is first
        assert dicts[0]["role"] == "system"
        assert "helpful AI familiar" in dicts[0]["content"]
        assert "cheerful wizard" in dicts[0]["content"]
        assert "favorite color is blue" in dicts[0]["content"]
        assert "discussing hobbies" in dicts[0]["content"]

        # Three distinct users follow
        user_messages = [d for d in dicts if d["role"] == "user"]
        assert len(user_messages) == 3
        assert [d["name"] for d in user_messages] == ["Alice", "Bob", "Charlie"]


# --- Retry logic ---


@pytest.fixture(autouse=True)
def _reset_semaphore() -> None:
    """Reset the module-level semaphore between tests."""
    llm_module._request_semaphore = None


class TestRetryLogic:
    @pytest.fixture
    def client(self) -> LLMClient:
        return LLMClient(api_key="test-key", model="openai/gpt-4o")

    @pytest.mark.asyncio
    async def test_post_retries_on_429_then_succeeds(self, client: LLMClient) -> None:
        """_post() retries once on 429, then returns the successful response."""
        resp_429 = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        resp_200 = httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", "http://x"),
        )
        mock_post = AsyncMock(side_effect=[resp_429, resp_200])

        with (
            patch.object(client, "_get_http") as mock_get_http,
            patch(
                "familiar_connect.llm.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            mock_get_http.return_value.post = mock_post
            result = await client._post("http://x", {}, {})

        assert result.status_code == 200
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_respects_retry_after_header(self, client: LLMClient) -> None:
        """_post() uses the Retry-After header value as sleep delay."""
        resp_429 = httpx.Response(
            429,
            headers={"Retry-After": "2"},
            request=httpx.Request("POST", "http://x"),
        )
        resp_200 = httpx.Response(
            200, json={}, request=httpx.Request("POST", "http://x")
        )
        mock_post = AsyncMock(side_effect=[resp_429, resp_200])

        with (
            patch.object(client, "_get_http") as mock_get_http,
            patch(
                "familiar_connect.llm.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            mock_get_http.return_value.post = mock_post
            await client._post("http://x", {}, {})

        mock_sleep.assert_called_once_with(2.0)

    @pytest.mark.asyncio
    async def test_post_gives_up_after_max_retries(self, client: LLMClient) -> None:
        """_post() returns the 429 response after exhausting retries."""
        resp_429 = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        mock_post = AsyncMock(return_value=resp_429)

        with (
            patch.object(client, "_get_http") as mock_get_http,
            patch(
                "familiar_connect.llm.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            mock_get_http.return_value.post = mock_post
            result = await client._post("http://x", {}, {})

        assert result.status_code == 429
        assert mock_sleep.call_count == _MAX_RETRIES

    @pytest.mark.asyncio
    async def test_post_does_not_retry_non_429(self, client: LLMClient) -> None:
        """_post() returns non-429 errors immediately without retrying."""
        resp_500 = httpx.Response(500, request=httpx.Request("POST", "http://x"))
        mock_post = AsyncMock(return_value=resp_500)

        with (
            patch.object(client, "_get_http") as mock_get_http,
            patch(
                "familiar_connect.llm.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            mock_get_http.return_value.post = mock_post
            result = await client._post("http://x", {}, {})

        assert result.status_code == 500
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_caps_backoff_at_max_delay(self, client: LLMClient) -> None:
        """Exponential backoff never exceeds _MAX_DELAY_S."""
        resp_429 = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        mock_post = AsyncMock(return_value=resp_429)

        with (
            patch.object(client, "_get_http") as mock_get_http,
            patch(
                "familiar_connect.llm.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            mock_get_http.return_value.post = mock_post
            await client._post("http://x", {}, {})

        for call in mock_sleep.call_args_list:
            assert call.args[0] <= _MAX_DELAY_S


# --- Connection pooling ---


class TestConnectionPooling:
    @pytest.mark.asyncio
    async def test_reuses_http_client_across_calls(self) -> None:
        """_get_http() returns the same client instance on repeated calls."""
        client = LLMClient(api_key="test-key", model="openai/gpt-4o")
        http1 = client._get_http()
        http2 = client._get_http()
        assert http1 is http2
        await client.close()

    @pytest.mark.asyncio
    async def test_close_shuts_down_http_client(self) -> None:
        """close() calls aclose() on the underlying httpx client."""
        client = LLMClient(api_key="test-key", model="openai/gpt-4o")
        http = client._get_http()
        with patch.object(http, "aclose", new_callable=AsyncMock) as mock_aclose:
            await client.close()
        mock_aclose.assert_called_once()


# --- Concurrency semaphore ---


class TestConcurrencySemaphore:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_requests(self) -> None:
        """At most 4 requests are in-flight simultaneously."""
        max_concurrent_seen = 0
        current = 0
        lock = asyncio.Lock()

        async def slow_post(*_args: object, **_kwargs: object) -> httpx.Response:
            nonlocal max_concurrent_seen, current
            async with lock:
                current += 1
                max_concurrent_seen = max(max_concurrent_seen, current)
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return httpx.Response(
                200, json={}, request=httpx.Request("POST", "http://x")
            )

        client = LLMClient(api_key="test-key", model="openai/gpt-4o")
        with patch.object(client, "_get_http") as mock_get_http:
            mock_get_http.return_value.post = slow_post
            tasks = [client._post("http://x", {}, {}) for _ in range(8)]
            await asyncio.gather(*tasks)

        assert max_concurrent_seen <= 4
        await client.close()


# --- Live integration test (requires OPENROUTER_API_KEY in env) ---

_has_api_key = bool(os.environ.get("OPENROUTER_API_KEY"))

_LIVE_MODEL = "mistralai/mistral-small-2603"


def _live_client() -> LLMClient:
    """Build a real :class:`LLMClient` from the process environment.

    The live test bypasses :func:`create_llm_clients` because it
    needs to run without a parsed :class:`CharacterConfig` in hand.
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    return LLMClient(api_key=api_key, model=_LIVE_MODEL)


@pytest.mark.integration
@pytest.mark.skipif(not _has_api_key, reason="OPENROUTER_API_KEY not set")
class TestOpenRouterLive:
    @pytest.mark.asyncio
    async def test_live_chat_returns_response(self) -> None:
        """Send a real request to OpenRouter and verify the response shape."""
        client = _live_client()
        messages = [
            Message(role="system", content="Reply with only the word 'pong'."),
            Message(role="user", content="ping", name="Tester"),
        ]
        result = await client.chat(messages)

        assert result.role == "assistant"
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_live_multi_user_chat(self) -> None:
        """Send a multi-user conversation to OpenRouter and get a reply."""
        client = _live_client()
        messages = [
            Message(
                role="system",
                content="You are a friendly familiar. Reply in one short sentence.",
            ),
            Message(role="user", content="I like cats.", name="Alice"),
            Message(role="user", content="I prefer dogs.", name="Bob"),
            Message(
                role="user",
                content="Familiar, who do you agree with?",
                name="Charlie",
            ),
        ]
        result = await client.chat(messages)

        assert result.role == "assistant"
        assert len(result.content) > 0
