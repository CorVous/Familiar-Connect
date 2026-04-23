"""Runtime bundle for the single active character.

One process runs one character (selected by ``FAMILIAR_ID``).
:class:`Familiar` carries config, history store, LLM client, and
subscriptions.

- :meth:`load_from_disk` — sole constructor; walks ``data/familiars/<id>/``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from familiar_connect.config import load_character_config
from familiar_connect.history.store import HistoryStore
from familiar_connect.subscriptions import SubscriptionRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.config import CharacterConfig
    from familiar_connect.llm import LLMClient
    from familiar_connect.transcription import DeepgramTranscriber
    from familiar_connect.tts import AzureTTSClient, CartesiaTTSClient, GeminiTTSClient


@dataclass
class Familiar:
    """Runtime bundle for one character.

    :param transcriber: when ``None``, voice subscription joins for TTS
        playback only — no incoming audio is transcribed.
    """

    id: str
    root: Path
    config: CharacterConfig
    history_store: HistoryStore
    llm_clients: dict[str, LLMClient]
    tts_client: CartesiaTTSClient | AzureTTSClient | GeminiTTSClient | None
    transcriber: DeepgramTranscriber | None
    subscriptions: SubscriptionRegistry
    bot_user_id: int | None = None
    """Discord snowflake for the logged-in bot user."""

    @classmethod
    def load_from_disk(
        cls,
        root: Path,
        *,
        llm_clients: dict[str, LLMClient],
        tts_client: CartesiaTTSClient | AzureTTSClient | GeminiTTSClient | None = None,
        transcriber: DeepgramTranscriber | None = None,
        defaults_path: Path | None = None,
    ) -> Familiar:
        """Build :class:`Familiar` from the on-disk ``data/familiars/<id>/`` layout.

        :param defaults_path: override for default profile path; tests
            pass this to avoid staging a sibling ``_default/`` folder.
        """
        familiar_id = root.name
        if defaults_path is None:
            defaults_path = root.parent / "_default" / "character.toml"
        character_config = load_character_config(
            root / "character.toml",
            defaults_path=defaults_path,
        )

        history_store = HistoryStore(root / "history.db")
        subscriptions = SubscriptionRegistry(root / "subscriptions.toml")

        return cls(
            id=familiar_id,
            root=root,
            config=character_config,
            history_store=history_store,
            llm_clients=llm_clients,
            tts_client=tts_client,
            transcriber=transcriber,
            subscriptions=subscriptions,
        )
