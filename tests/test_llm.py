"""Tests for the LLM client (OpenRouter integration)."""

from __future__ import annotations

import asyncio
import json
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
        """SystemPromptLayers holds every system-prompt section."""
        layers = SystemPromptLayers(
            character_card="You are Merlin the wizard.",
            rag_context="User Alice likes cats.",
            conversation_summary="They discussed pets.",
            recent_history=[],
        )
        assert layers.character_card == "You are Merlin the wizard."
        assert layers.rag_context == "User Alice likes cats."
        assert layers.conversation_summary == "They discussed pets."
        assert layers.recent_history == []

    def test_build_system_prompt_ordering(self) -> None:
        """System prompt sections appear in the correct order."""
        layers = SystemPromptLayers(
            character_card="CHARACTER",
            rag_context="RAG",
            conversation_summary="SUMMARY",
            recent_history=[],
        )
        prompt = build_system_prompt(layers)
        char_pos = prompt.index("CHARACTER")
        rag_pos = prompt.index("RAG")
        summary_pos = prompt.index("SUMMARY")
        assert char_pos < rag_pos < summary_pos

    def test_build_system_prompt_omits_empty_sections(self) -> None:
        """Empty optional layers are excluded from the system prompt."""
        layers = SystemPromptLayers(
            character_card="CHARACTER",
            rag_context="",
            conversation_summary="",
            recent_history=[],
        )
        prompt = build_system_prompt(layers)
        assert "CHARACTER" in prompt
        # Empty sections should not leave stray headers or blank blocks
        assert prompt.count("\n\n\n") == 0


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

    def test_payload_omits_provider_when_unset(self) -> None:
        """Default routing — no ``provider`` field in the request."""
        client = LLMClient(api_key="k", model="m")
        payload = client.build_payload([Message(role="user", content="x")])
        assert "provider" not in payload

    def test_payload_pins_provider_order(self) -> None:
        """``provider_order`` becomes ``provider.order`` for OpenRouter."""
        client = LLMClient(
            api_key="k",
            model="m",
            provider_order=("z-ai", "deepinfra"),
        )
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["provider"] == {
            "order": ["z-ai", "deepinfra"],
            "allow_fallbacks": True,
        }

    def test_payload_disables_fallbacks_when_requested(self) -> None:
        client = LLMClient(
            api_key="k",
            model="m",
            provider_order=("z-ai",),
            provider_allow_fallbacks=False,
        )
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["provider"]["allow_fallbacks"] is False

    def test_payload_omits_reasoning_when_unset(self) -> None:
        """Default — no ``reasoning`` field; defer to model default."""
        client = LLMClient(api_key="k", model="m")
        payload = client.build_payload([Message(role="user", content="x")])
        assert "reasoning" not in payload

    def test_payload_reasoning_off_excludes(self) -> None:
        """``reasoning="off"`` maps to OpenRouter ``reasoning.exclude=True``."""
        client = LLMClient(api_key="k", model="m", reasoning="off")
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["reasoning"] == {"exclude": True}

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_payload_reasoning_effort(self, level: str) -> None:
        """Effort levels map to OpenRouter ``reasoning.effort``."""
        client = LLMClient(api_key="k", model="m", reasoning=level)
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["reasoning"] == {"effort": level}

    def test_payload_omits_sampling_params_when_unset(self) -> None:
        """Default — no sampling fields beyond temperature; provider defaults."""
        client = LLMClient(api_key="k", model="m")
        payload = client.build_payload([Message(role="user", content="x")])
        assert "top_p" not in payload
        assert "top_k" not in payload
        assert "presence_penalty" not in payload

    def test_payload_includes_sampling_params(self) -> None:
        """top_p / top_k / presence_penalty pass through to the payload."""
        client = LLMClient(
            api_key="k",
            model="m",
            top_p=0.95,
            top_k=20,
            presence_penalty=1.5,
        )
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["top_p"] == pytest.approx(0.95)
        assert payload["top_k"] == 20
        assert payload["presence_penalty"] == pytest.approx(1.5)

    def test_payload_reasoning_max_tokens(self) -> None:
        """``reasoning_max_tokens`` maps to OpenRouter ``reasoning.max_tokens``."""
        client = LLMClient(api_key="k", model="m", reasoning_max_tokens=2048)
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["reasoning"] == {"max_tokens": 2048}

    def test_payload_reasoning_max_tokens_wins_over_effort(self) -> None:
        """OpenRouter accepts one of effort/max_tokens — explicit budget wins."""
        client = LLMClient(
            api_key="k", model="m", reasoning="low", reasoning_max_tokens=2048
        )
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["reasoning"] == {"max_tokens": 2048}

    def test_payload_no_prefill_by_default(self) -> None:
        client = LLMClient(api_key="k", model="m")
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["messages"][-1]["role"] == "user"

    def test_payload_think_prepend_appends_prefill(self) -> None:
        """Qwen3 no-think stabiliser: fake closed think block as final message."""
        client = LLMClient(api_key="k", model="m", think_prepend=True)
        payload = client.build_payload([Message(role="user", content="x")])
        assert payload["messages"][-1] == {
            "role": "assistant",
            "content": "<think>\n\n</think>",
        }
        # original user message still present before the prefill
        assert payload["messages"][-2]["role"] == "user"

    def test_payload_anthropic_caches_system_prompt(self) -> None:
        """Anthropic models get a cache_control breakpoint on the system block."""
        client = LLMClient(api_key="k", model="anthropic/claude-haiku-4.5")
        messages = [
            Message(role="system", content="You are a wizard familiar."),
            Message(role="user", content="Hi", name="Alice"),
        ]
        payload = client.build_payload(messages)
        system = payload["messages"][0]
        assert system["role"] == "system"
        assert isinstance(system["content"], list)
        last_block = system["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}
        assert last_block["text"] == "You are a wizard familiar."

    def test_payload_non_anthropic_leaves_system_string(self) -> None:
        """Non-Anthropic models keep a plain-string system prompt, no caching."""
        client = LLMClient(api_key="k", model="z-ai/glm-5.1")
        messages = [
            Message(role="system", content="You are a wizard familiar."),
            Message(role="user", content="Hi", name="Alice"),
        ]
        payload = client.build_payload(messages)
        assert payload["messages"][0]["content"] == "You are a wizard familiar."
        assert "cache_control" not in json.dumps(payload)

    def test_payload_caching_leaves_user_assistant_unchanged(self) -> None:
        """Cache breakpoint touches only the system message, never user/assistant."""
        messages = [
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hi", name="Alice"),
            Message(role="assistant", content="Hello!"),
        ]
        for model in ("anthropic/claude-haiku-4.5", "z-ai/glm-5.1"):
            client = LLMClient(api_key="k", model=model)
            payload = client.build_payload(messages)
            assert payload["messages"][1] == messages[1].to_dict()
            assert payload["messages"][2] == messages[2].to_dict()


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
    async def test_chat_logs_response_body_on_4xx(
        self,
        client: LLMClient,
        sample_messages: list[Message],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A 400 leaves the upstream error body on a WARNING log."""
        body = (
            '{"error":{"message":"Unsupported value: '
            "'temperature' does not support 0.7 with this model.\","
            '"type":"invalid_request_error"}}'
        )
        error = httpx.HTTPStatusError(
            "Bad Request",
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
            response=httpx.Response(400, text=body),
        )
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.text = body
        mock_response.raise_for_status.side_effect = error

        caplog.set_level("WARNING", logger="familiar_connect.llm")
        with (
            patch.object(client, "_post", return_value=mock_response),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await client.chat(sample_messages)

        joined = "\n".join(rec.message for rec in caplog.records)
        assert "400" in joined
        assert "temperature" in joined

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
        """User config overrides the prose slot from the default profile."""
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.prose]\nmodel = "user/custom-prose"\ntemperature = 0.9\n',
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        clients = create_llm_clients("sk-or-test-abc", cfg)
        assert clients["prose"].model == "user/custom-prose"
        assert clients["prose"].temperature == pytest.approx(0.9)

    def test_reasoning_threaded_through(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """Slot reasoning lands on its :class:`LLMClient`."""
        cfg = load_character_config(
            tmp_path / "missing.toml",
            defaults_path=default_profile_path,
        )
        clients = create_llm_clients("sk-or-test-abc", cfg)
        # default profile: fast="off", prose+background="medium"
        assert clients["fast"].reasoning == "off"
        assert clients["prose"].reasoning == "medium"
        assert clients["background"].reasoning == "medium"

    def test_sampling_and_think_prepend_threaded_through(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """Slot sampling knobs + think_prepend land on its :class:`LLMClient`."""
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.fast]\nmodel = "qwen/qwen3.6-35b-a3b"\ntop_p = 0.8\n'
            "top_k = 20\npresence_penalty = 1.5\nthink_prepend = true\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        clients = create_llm_clients("sk-or-test-abc", cfg)
        fast = clients["fast"]
        assert fast.top_p == pytest.approx(0.8)
        assert fast.top_k == 20
        assert fast.presence_penalty == pytest.approx(1.5)
        assert fast.think_prepend is True
        # untouched slot keeps provider defaults
        assert clients["prose"].top_p is None
        assert clients["prose"].think_prepend is False

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

    def test_builds_description_client_when_configured(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """``image_description_model`` at [llm] level produces reserved client."""
        path = tmp_path / "character.toml"
        path.write_text('[llm]\nimage_description_model = "openai/gpt-4o"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        clients = create_llm_clients("sk-or-test-abc", cfg)
        assert "__image_description__" in clients
        assert clients["__image_description__"].model == "openai/gpt-4o"

    def test_no_description_client_when_not_configured(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """Without ``image_description_model``, reserved key absent."""
        cfg = load_character_config(
            tmp_path / "missing.toml",
            defaults_path=default_profile_path,
        )
        clients = create_llm_clients("sk-or-test-abc", cfg)
        assert "__image_description__" not in clients

    def test_image_tools_multimodal_threaded_through(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """image_tools + multimodal slot flags land on LLMClient."""
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.prose]\nmodel = "x/y"\nimage_tools = true\nmultimodal = true\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        clients = create_llm_clients("sk-or-test-abc", cfg)
        assert clients["prose"].image_tools_enabled is True
        assert clients["prose"].multimodal is True


# --- Integration-style test with full prompt assembly ---


class TestPromptAssembly:
    def test_full_prompt_assembly_with_multi_user(self) -> None:
        """Assemble a complete message list: system prompt + multi-user history."""
        layers = SystemPromptLayers(
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


# --- Configured request-concurrency cap ---


class TestRequestSemaphoreConfig:
    def test_configure_request_semaphore_sets_cap(self) -> None:
        sem = llm_module.configure_request_semaphore(7)
        assert llm_module.get_request_semaphore() is sem
        assert sem._value == 7

    def test_create_llm_clients_applies_configured_cap(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """``[llm].max_concurrent_requests`` sizes the shared semaphore."""
        path = tmp_path / "character.toml"
        path.write_text("[llm]\nmax_concurrent_requests = 6\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        create_llm_clients("sk-or-test-abc", cfg)
        assert llm_module.get_request_semaphore()._value == 6
