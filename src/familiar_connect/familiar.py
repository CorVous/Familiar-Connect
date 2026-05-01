"""Runtime bundle for the single active character.

One process runs one character (selected by ``FAMILIAR_ID``).
:class:`Familiar` carries config, history store, LLM client, bus,
router, and subscriptions.

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
    from familiar_connect.transcription import DeepgramTranscriber
    from familiar_connect.tts import AzureTTSClient, CartesiaTTSClient, GeminiTTSClient
    from familiar_connect.voice.turn_detection import LocalTurnDetector


@dataclass
class Familiar:
    """Runtime bundle for one character.

    :param transcriber: when ``None``, voice subscription joins for TTS
        playback only — no incoming audio is transcribed.
    :param bus: event bus — sources publish, processors consume.
    :param router: per-session turn routing + cancel-prior-scope.
    """

    id: str
    root: Path
    config: CharacterConfig
    history_store: HistoryStore
    llm_clients: dict[str, LLMClient]
    tts_client: CartesiaTTSClient | AzureTTSClient | GeminiTTSClient | None
    transcriber: DeepgramTranscriber | None
    subscriptions: SubscriptionRegistry
    bus: EventBus = field(default_factory=InProcessEventBus)
    router: TurnRouter = field(default_factory=TurnRouter)
    bot_user_id: int | None = None
    """Discord snowflake for the logged-in bot user."""
    local_turn_detector: LocalTurnDetector | None = None
    """V1 phase 2: Silero VAD + Smart Turn local endpointer factory.

    When set, the voice intake forks per-user PCM into both Deepgram
    and a local detector chain; turn-complete decisions trigger
    ``transcriber.finalize()`` so Deepgram emits its final ahead of
    its own silence timer. ``None`` keeps Deepgram's hosted endpointer
    in charge.
    """

    @classmethod
    def load_from_disk(
        cls,
        root: Path,
        *,
        llm_clients: dict[str, LLMClient],
        tts_client: CartesiaTTSClient | AzureTTSClient | GeminiTTSClient | None = None,
        transcriber: DeepgramTranscriber | None = None,
        local_turn_detector: LocalTurnDetector | None = None,
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
            local_turn_detector=local_turn_detector,
        )
