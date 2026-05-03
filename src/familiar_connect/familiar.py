"""Runtime bundle for the single active character.

One process per character (selected by ``FAMILIAR_ID``).
:class:`Familiar` carries config, history store, LLM client, bus,
router, subscriptions.

- :meth:`load_from_disk` — sole constructor; walks ``data/familiars/<id>/``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect.bus import InProcessEventBus, TurnRouter
from familiar_connect.config import load_character_config
from familiar_connect.history.store import HistoryStore
from familiar_connect.subscriptions import SubscriptionRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.config import CharacterConfig
    from familiar_connect.llm import LLMClient
    from familiar_connect.stt import Transcriber
    from familiar_connect.tts import AzureTTSClient, CartesiaTTSClient, GeminiTTSClient
    from familiar_connect.voice.turn_detection import LocalTurnDetector


@dataclass
class Familiar:
    """Runtime bundle for one character.

    transcriber: ``None`` = voice subscription joins for TTS playback
    only; incoming audio not transcribed.
    bus: sources publish, processors consume.
    router: per-session turn routing + cancel-prior-scope.
    """

    id: str
    root: Path
    config: CharacterConfig
    history_store: HistoryStore
    llm_clients: dict[str, LLMClient]
    tts_client: CartesiaTTSClient | AzureTTSClient | GeminiTTSClient | None
    transcriber: Transcriber | None
    subscriptions: SubscriptionRegistry
    bus: EventBus = field(default_factory=InProcessEventBus)
    router: TurnRouter = field(default_factory=TurnRouter)
    bot_user_id: int | None = None
    """Discord snowflake for the logged-in bot user."""
    local_turn_detector: LocalTurnDetector | None = None
    """V1 phase 2: TEN-VAD + Smart Turn local endpointer factory.

    When set, voice intake forks per-user PCM to Deepgram + local chain;
    turn-complete decisions fire ``transcriber.finalize()`` so Deepgram
    emits final ahead of its silence timer. ``None`` leaves Deepgram's
    hosted endpointer in charge.
    """

    @classmethod
    def load_from_disk(
        cls,
        root: Path,
        *,
        llm_clients: dict[str, LLMClient],
        tts_client: CartesiaTTSClient | AzureTTSClient | GeminiTTSClient | None = None,
        transcriber: Transcriber | None = None,
        local_turn_detector: LocalTurnDetector | None = None,
        defaults_path: Path | None = None,
    ) -> Familiar:
        """Build from on-disk ``data/familiars/<id>/`` layout.

        defaults_path: override default profile path; tests use this to
        skip staging a sibling ``_default/`` folder.
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
            local_turn_detector=local_turn_detector,
        )
