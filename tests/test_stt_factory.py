"""STT backend selector — `[providers.stt].backend` dispatch."""

from __future__ import annotations

import os

import pytest

from familiar_connect.config import (
    DeepgramSTTConfig,
    FasterWhisperSTTConfig,
    ParakeetSTTConfig,
    STTConfig,
)
from familiar_connect.stt import Transcriber, create_transcriber
from familiar_connect.stt.deepgram import DeepgramTranscriber
from familiar_connect.stt.faster_whisper import FasterWhisperTranscriber
from familiar_connect.stt.parakeet import ParakeetTranscriber


@pytest.fixture(autouse=True)
def _scrub_env() -> None:
    """Drop the one secret the Deepgram factory still consumes from env."""
    os.environ.pop("DEEPGRAM_API_KEY", None)


class TestProtocolContract:
    def test_deepgram_satisfies_transcriber_protocol(self) -> None:
        t = DeepgramTranscriber(api_key="x")
        assert isinstance(t, Transcriber)

    def test_parakeet_satisfies_transcriber_protocol(self) -> None:
        t = ParakeetTranscriber()
        assert isinstance(t, Transcriber)

    def test_faster_whisper_satisfies_transcriber_protocol(self) -> None:
        t = FasterWhisperTranscriber()
        assert isinstance(t, Transcriber)


class TestDispatch:
    def test_deepgram_backend_returns_deepgram_transcriber(self) -> None:
        os.environ["DEEPGRAM_API_KEY"] = "test-key"
        cfg = STTConfig(backend="deepgram", deepgram=DeepgramSTTConfig())

        t = create_transcriber(cfg)

        assert isinstance(t, DeepgramTranscriber)
        assert t.api_key == "test-key"

    def test_parakeet_backend_returns_parakeet_transcriber(self) -> None:
        cfg = STTConfig(backend="parakeet", parakeet=ParakeetSTTConfig())

        t = create_transcriber(cfg)

        assert isinstance(t, ParakeetTranscriber)
        assert t.model_name == "nvidia/parakeet-tdt-0.6b-v3"

    def test_faster_whisper_backend_returns_faster_whisper_transcriber(self) -> None:
        cfg = STTConfig(
            backend="faster_whisper", faster_whisper=FasterWhisperSTTConfig()
        )

        t = create_transcriber(cfg)

        assert isinstance(t, FasterWhisperTranscriber)
        assert t.model_size == "small"

    def test_missing_api_key_raises_value_error(self) -> None:
        # mirrors prior `create_transcriber_from_env` contract — caller catches
        cfg = STTConfig(backend="deepgram")
        with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
            create_transcriber(cfg)


class TestUnknownBackend:
    def test_unknown_backend_rejected(self) -> None:
        # ``_parse_stt_config`` whitelists at parse time, but the dataclass
        # itself doesn't — exercise the factory's own dispatch guard.
        cfg = STTConfig(backend="vosk", deepgram=DeepgramSTTConfig())

        with pytest.raises(ValueError, match="vosk"):
            create_transcriber(cfg)
