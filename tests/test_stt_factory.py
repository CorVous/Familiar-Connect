"""STT backend selector — `[providers.stt].backend` + ``STT_BACKEND``."""

from __future__ import annotations

import os

import pytest

from familiar_connect.config import DeepgramSTTConfig, STTConfig
from familiar_connect.stt import Transcriber, create_transcriber
from familiar_connect.stt.deepgram import DeepgramTranscriber


@pytest.fixture(autouse=True)
def _scrub_env() -> None:
    """Drop env that perturbs selector + Deepgram factory between tests."""
    for k in ("STT_BACKEND", "DEEPGRAM_API_KEY"):
        os.environ.pop(k, None)


class TestProtocolContract:
    def test_deepgram_satisfies_transcriber_protocol(self) -> None:
        t = DeepgramTranscriber(api_key="x")
        assert isinstance(t, Transcriber)


class TestDispatch:
    def test_deepgram_backend_returns_deepgram_transcriber(self) -> None:
        os.environ["DEEPGRAM_API_KEY"] = "test-key"
        cfg = STTConfig(backend="deepgram", deepgram=DeepgramSTTConfig())

        t = create_transcriber(cfg)

        assert isinstance(t, DeepgramTranscriber)
        assert t.api_key == "test-key"

    def test_missing_api_key_raises_value_error(self) -> None:
        # mirrors prior `create_transcriber_from_env` contract — caller catches
        cfg = STTConfig(backend="deepgram")
        with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
            create_transcriber(cfg)


class TestEnvOverride:
    def test_stt_backend_env_overrides_toml(self) -> None:
        # TOML says deepgram; env points at an unknown backend → ValueError
        # (positive proof env wins; once parakeet/faster_whisper land, swap to
        # an asserting-isinstance check)
        os.environ["STT_BACKEND"] = "parakeet"
        os.environ["DEEPGRAM_API_KEY"] = "test-key"
        cfg = STTConfig(backend="deepgram")

        with pytest.raises(ValueError, match="parakeet"):
            create_transcriber(cfg)

    def test_unknown_toml_backend_rejected(self) -> None:
        # ``_parse_stt_config`` whitelists at parse time, but the dataclass
        # itself doesn't — exercise the factory's own dispatch guard.
        cfg = STTConfig(backend="whisper_local", deepgram=DeepgramSTTConfig())

        with pytest.raises(ValueError, match="whisper_local"):
            create_transcriber(cfg)
