"""Tests for the LLM client (OpenRouter integration)."""

import os
from unittest.mock import Mock, patch

import httpx
import pytest

from familiar_connect.llm import (
    LLMClient,
    Message,
    SystemPromptLayers,
    build_system_prompt,
    create_client_from_env,
)

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
    def test_init_requires_api_key(self) -> None:
        """Client requires an API key."""
        client = LLMClient(api_key="test-key")
        assert client.api_key == "test-key"

    def test_init_default_model(self) -> None:
        """Client has a sensible default model."""
        client = LLMClient(api_key="test-key")
        assert client.model is not None
        assert isinstance(client.model, str)

    def test_init_custom_model(self) -> None:
        """Client accepts a custom model override."""
        client = LLMClient(api_key="test-key", model="anthropic/claude-sonnet-4")
        assert client.model == "anthropic/claude-sonnet-4"

    def test_init_default_base_url(self) -> None:
        """Client defaults to OpenRouter's API URL."""
        client = LLMClient(api_key="test-key")
        assert "openrouter.ai" in client.base_url

    def test_init_custom_base_url(self) -> None:
        """Client accepts a custom base URL for testing or alternative endpoints."""
        client = LLMClient(api_key="test-key", base_url="https://custom.api/v1")
        assert client.base_url == "https://custom.api/v1"

    def test_builds_request_headers(self) -> None:
        """Request headers include Authorization and required OpenRouter headers."""
        client = LLMClient(api_key="sk-or-test-123")
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
        client = LLMClient(api_key="test-key", temperature=0.7)
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


# --- Factory from environment ---


class TestCreateClientFromEnv:
    def test_creates_client_from_env_vars(self) -> None:
        """Factory reads OPENROUTER_API_KEY and optional overrides from env."""
        env = {
            "OPENROUTER_API_KEY": "sk-or-test-abc",
            "OPENROUTER_MODEL": "anthropic/claude-sonnet-4",
            "OPENROUTER_TEMPERATURE": "0.5",
        }
        with patch.dict(os.environ, env, clear=False):
            client = create_client_from_env()

        assert client.api_key == "sk-or-test-abc"
        assert client.model == "anthropic/claude-sonnet-4"
        assert client.temperature == pytest.approx(0.5)

    def test_uses_defaults_when_optional_vars_missing(self) -> None:
        """Factory uses default model and no temperature when env vars are absent."""
        env = {"OPENROUTER_API_KEY": "sk-or-test-abc"}
        with patch.dict(os.environ, env, clear=False):
            # Remove optional vars if they happen to be set
            os.environ.pop("OPENROUTER_MODEL", None)
            os.environ.pop("OPENROUTER_TEMPERATURE", None)
            client = create_client_from_env()

        assert client.api_key == "sk-or-test-abc"
        assert "openrouter.ai" in client.base_url
        assert client.temperature is None

    def test_raises_when_api_key_missing(self) -> None:
        """Factory raises a clear error when OPENROUTER_API_KEY is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match=r"OPENROUTER_API_KEY"),
        ):
            create_client_from_env()


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


# --- Live integration test (requires OPENROUTER_API_KEY in env) ---

_has_api_key = bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.skipif(not _has_api_key, reason="OPENROUTER_API_KEY not set")
class TestOpenRouterLive:
    @pytest.mark.asyncio
    async def test_live_chat_returns_response(self) -> None:
        """Send a real request to OpenRouter and verify the response shape."""
        client = create_client_from_env()
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
        client = create_client_from_env()
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
