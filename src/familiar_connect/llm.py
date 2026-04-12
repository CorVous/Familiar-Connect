"""LLM client for generating AI responses via OpenRouter."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from familiar_connect.config import LLM_SLOT_NAMES

if TYPE_CHECKING:
    from typing import Any, Self

    from familiar_connect.config import CharacterConfig

_logger = logging.getLogger(__name__)

_NAME_ALLOWED = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_name(name: str) -> str | None:
    """Return a name safe for the OpenAI name field, or None if empty after sanitizing.

    The API requires names matching ^[a-zA-Z0-9_-]{1,64}$.
    Spaces and unsupported chars are replaced with underscores.
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
    """Return the module-level semaphore, creating it on first call.

    Must be called inside a running event loop. The semaphore is shared
    across all :class:`LLMClient` instances because the bottleneck is
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
    """The five layers that compose the system prompt per the plan.

    1. Core instructions and safety rails
    2. Character card (personality, speech patterns, backstory)
    3. Retrieved RAG context
    4. Conversation summary
    5. Recent message history (kept separate as messages, not in system prompt)
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
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._http: httpx.AsyncClient | None = None

    def _get_http(self: Self) -> httpx.AsyncClient:
        """Return the shared HTTP client, creating it lazily."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=120.0)
        return self._http

    async def close(self: Self) -> None:
        """Shut down the underlying HTTP connection pool."""
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

            # Last attempt — don't sleep, just return the 429.
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

        # All retries exhausted — caller's raise_for_status() will handle it.
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


def create_llm_clients(
    api_key: str,
    character_config: CharacterConfig,
) -> dict[str, LLMClient]:
    """Build one :class:`LLMClient` per call-site slot on *character_config*.

    Every slot in :data:`familiar_connect.config.LLM_SLOT_NAMES` must
    appear in ``character_config.llm`` (the loader guarantees this by
    merging over ``_default/character.toml``). The returned dict is
    keyed by slot name and each client carries the slot's own
    ``model`` and ``temperature``.

    All clients share the same API key and the process-wide rate-limit
    semaphore in :func:`get_request_semaphore`, so splitting one
    ``side`` client into five does not multiply concurrency against
    the OpenRouter rate limit.

    :param api_key: ``OPENROUTER_API_KEY`` value. Callers pull this
        from the environment themselves so the factory stays pure.
    :param character_config: Parsed :class:`CharacterConfig` whose
        ``llm`` dict holds one :class:`LLMSlotConfig` per slot.
    :return: ``slot_name -> LLMClient`` map, one entry per slot name.
    """
    clients: dict[str, LLMClient] = {}
    for slot_name in LLM_SLOT_NAMES:
        slot = character_config.llm[slot_name]
        clients[slot_name] = LLMClient(
            api_key=api_key,
            model=slot.model,
            temperature=slot.temperature,
        )
        _logger.info(
            "LLM slot %s -> model=%s temperature=%s",
            slot_name,
            slot.model,
            slot.temperature,
        )
    return clients
