"""Tests for the memory-projector registry (M5).

Lifts existing writers (`SummaryWorker`, `FactExtractor`,
`PeopleDossierWorker`, `ReflectionWorker`) behind a
``MemoryProjector`` Protocol selectable via
``[providers.memory].projectors``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.config import (
    FactSupersedeConfig,
    MemoryProvidersConfig,
    PeopleDossierConfig,
    ReflectionConfig,
    RichNoteConfig,
    RollingSummaryConfig,
)
from familiar_connect.embedding import HashEmbedder
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors import projectors as projectors_module
from familiar_connect.processors.fact_embedding_worker import FactEmbeddingWorker
from familiar_connect.processors.fact_extractor import FactExtractor
from familiar_connect.processors.fact_supersede_worker import FactSupersedeWorker
from familiar_connect.processors.people_dossier_worker import PeopleDossierWorker
from familiar_connect.processors.projectors import (
    DEFAULT_PROJECTORS,
    ProjectorContext,
    create_projectors,
    known_projectors,
    register_projector,
)
from familiar_connect.processors.reflection_worker import ReflectionWorker
from familiar_connect.processors.summary_worker import SummaryWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


class _StubLLM(LLMClient):
    """No-op LLM stub; projector wiring tests don't drive real ticks."""

    def __init__(self) -> None:
        super().__init__(api_key="k", model="m")

    async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
        return Message(role="assistant", content="[]")

    async def chat_stream(  # type: ignore[override]
        self, messages: list[Message]
    ) -> AsyncIterator[str]:
        reply = await self.chat(messages)
        yield reply.content_str


def _ctx(*, embedder: HashEmbedder | None = None) -> ProjectorContext:
    store = HistoryStore(":memory:")
    return ProjectorContext(
        store=AsyncHistoryStore(store),
        llm_clients={"background": _StubLLM(), "fast": _StubLLM(), "prose": _StubLLM()},
        familiar_id="fam",
        embedder=embedder,
    )


class TestProjectorRegistry:
    def test_default_projectors_lists_five_builtins(self) -> None:
        assert set(DEFAULT_PROJECTORS) == {
            "rolling_summary",
            "rich_note",
            "people_dossier",
            "reflection",
            "fact_supersede",
        }

    def test_known_projectors_includes_all_builtins(self) -> None:
        assert {
            "rolling_summary",
            "rich_note",
            "people_dossier",
            "reflection",
            "fact_supersede",
        } <= known_projectors()

    def test_create_projectors_returns_instances_in_order(self) -> None:
        projectors = create_projectors(
            names=["rich_note", "rolling_summary"],
            context=_ctx(),
        )
        assert len(projectors) == 2
        # Order preserved from `names`.
        assert isinstance(projectors[0], FactExtractor)
        assert isinstance(projectors[1], SummaryWorker)

    def test_create_projectors_default_yields_all_builtins(self) -> None:
        projectors = create_projectors(
            names=list(DEFAULT_PROJECTORS),
            context=_ctx(),
        )
        types = {type(p) for p in projectors}
        assert types == {
            SummaryWorker,
            FactExtractor,
            PeopleDossierWorker,
            ReflectionWorker,
            FactSupersedeWorker,
        }

    def test_empty_names_yields_empty_list(self) -> None:
        assert create_projectors(names=[], context=_ctx()) == []

    def test_unknown_projector_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown memory projector"):
            create_projectors(names=["nonexistent"], context=_ctx())

    def test_register_projector_adds_to_registry(self) -> None:
        class _Custom:
            name = "custom"

            async def run(self) -> None:
                pass

        register_projector("custom_test_projector", lambda _ctx: _Custom())
        try:
            assert "custom_test_projector" in known_projectors()
            projectors = create_projectors(
                names=["custom_test_projector"],
                context=_ctx(),
            )
            assert len(projectors) == 1
            assert projectors[0].name == "custom"
        finally:
            # Restore registry — avoid leaking into other tests.
            projectors_module._REGISTRY.pop("custom_test_projector", None)

    def test_each_builtin_projector_implements_protocol(self) -> None:
        """Every default projector exposes ``name`` + async ``run``."""
        for name in DEFAULT_PROJECTORS:
            [proj] = create_projectors(names=[name], context=_ctx())
            assert isinstance(proj.name, str)
            assert proj.name
            assert callable(proj.run)

    def test_fact_embedding_known_but_not_default(self) -> None:
        """M6 stays opt-in — registered, but not in the default tuple."""
        assert "fact_embedding" in known_projectors()
        assert "fact_embedding" not in DEFAULT_PROJECTORS

    def test_fact_embedding_factory_requires_embedder(self) -> None:
        with pytest.raises(ValueError, match="fact_embedding projector requires"):
            create_projectors(names=["fact_embedding"], context=_ctx())

    def test_fact_embedding_factory_yields_worker_when_wired(self) -> None:
        embedder = HashEmbedder(dim=64)
        [proj] = create_projectors(
            names=["fact_embedding"],
            context=_ctx(embedder=embedder),
        )
        assert isinstance(proj, FactEmbeddingWorker)


class TestMemoryProvidersConfig:
    def test_default_projectors_in_character_config(self) -> None:
        from familiar_connect.config import CharacterConfig  # noqa: PLC0415

        cfg = CharacterConfig()
        assert tuple(cfg.memory_providers.projectors) == DEFAULT_PROJECTORS

    def test_load_with_explicit_projectors_list(self, tmp_path: Path) -> None:
        from familiar_connect.config import load_character_config  # noqa: PLC0415

        defaults = tmp_path / "_default" / "character.toml"
        defaults.parent.mkdir(parents=True)
        defaults.write_text(
            '[providers.memory]\nprojectors = ["rolling_summary", "rich_note"]\n',
            encoding="utf-8",
        )
        target = tmp_path / "fam" / "character.toml"
        target.parent.mkdir(parents=True)
        target.write_text("", encoding="utf-8")
        cfg = load_character_config(target, defaults_path=defaults)
        assert cfg.memory_providers.projectors == ("rolling_summary", "rich_note")

    def test_familiar_override_replaces_default_list(self, tmp_path: Path) -> None:
        from familiar_connect.config import load_character_config  # noqa: PLC0415

        defaults = tmp_path / "_default" / "character.toml"
        defaults.parent.mkdir(parents=True)
        defaults.write_text(
            "[providers.memory]\n"
            'projectors = ["rolling_summary", "rich_note", "people_dossier", '
            '"reflection"]\n',
            encoding="utf-8",
        )
        target = tmp_path / "fam" / "character.toml"
        target.parent.mkdir(parents=True)
        target.write_text(
            '[providers.memory]\nprojectors = ["rolling_summary"]\n',
            encoding="utf-8",
        )
        cfg = load_character_config(target, defaults_path=defaults)
        assert cfg.memory_providers.projectors == ("rolling_summary",)

    def test_invalid_projectors_type_rejected(self, tmp_path: Path) -> None:
        from familiar_connect.config import (  # noqa: PLC0415
            ConfigError,
            load_character_config,
        )

        defaults = tmp_path / "_default" / "character.toml"
        defaults.parent.mkdir(parents=True)
        defaults.write_text(
            "[providers.memory]\nprojectors = 'not-a-list'\n",
            encoding="utf-8",
        )
        target = tmp_path / "fam" / "character.toml"
        target.parent.mkdir(parents=True)
        target.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError, match="projectors"):
            load_character_config(target, defaults_path=defaults)

    def test_unknown_projector_name_rejected_at_load(self, tmp_path: Path) -> None:
        from familiar_connect.config import (  # noqa: PLC0415
            ConfigError,
            load_character_config,
        )

        defaults = tmp_path / "_default" / "character.toml"
        defaults.parent.mkdir(parents=True)
        defaults.write_text(
            "[providers.memory]\nprojectors = ['rolling_summary', 'nonexistent']\n",
            encoding="utf-8",
        )
        target = tmp_path / "fam" / "character.toml"
        target.parent.mkdir(parents=True)
        target.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown memory projector"):
            load_character_config(target, defaults_path=defaults)


class TestWorkerKnobThreading:
    """Factories thread ``[providers.memory.<name>]`` knobs into workers."""

    def _ctx_with(self, memory: MemoryProvidersConfig) -> ProjectorContext:
        store = HistoryStore(":memory:")
        return ProjectorContext(
            store=AsyncHistoryStore(store),
            llm_clients={"background": _StubLLM()},
            familiar_id="fam",
            memory=memory,
        )

    def test_context_defaults_to_default_memory_config(self) -> None:
        assert _ctx().memory == MemoryProvidersConfig()

    def test_rolling_summary_knobs(self) -> None:
        memory = MemoryProvidersConfig(
            rolling_summary=RollingSummaryConfig(
                turns_threshold=3, cross_k=2, tick_interval_s=1.5
            )
        )
        [w] = create_projectors(
            names=["rolling_summary"], context=self._ctx_with(memory)
        )
        assert isinstance(w, SummaryWorker)
        assert w._turns_threshold == 3
        assert w._cross_k == 2
        assert w._tick_interval_s == pytest.approx(1.5)

    def test_rich_note_knobs(self) -> None:
        memory = MemoryProvidersConfig(
            rich_note=RichNoteConfig(
                batch_size=3, tick_interval_s=7.0, participants_max=12
            )
        )
        [w] = create_projectors(names=["rich_note"], context=self._ctx_with(memory))
        assert isinstance(w, FactExtractor)
        assert w._batch_size == 3
        assert w._tick_interval_s == pytest.approx(7.0)
        assert w._participants_max == 12

    def test_people_dossier_knobs(self) -> None:
        memory = MemoryProvidersConfig(
            people_dossier=PeopleDossierConfig(tick_interval_s=11.0)
        )
        [w] = create_projectors(
            names=["people_dossier"], context=self._ctx_with(memory)
        )
        assert isinstance(w, PeopleDossierWorker)
        assert w._tick_interval_s == pytest.approx(11.0)

    def test_reflection_knobs(self) -> None:
        memory = MemoryProvidersConfig(
            reflection=ReflectionConfig(
                turns_threshold=8,
                max_reflections_per_tick=1,
                max_turns_per_tick=25,
                recent_facts_limit=5,
                tick_interval_s=90.0,
            )
        )
        [w] = create_projectors(names=["reflection"], context=self._ctx_with(memory))
        assert isinstance(w, ReflectionWorker)
        assert w._turns_threshold == 8
        assert w._max_per_tick == 1
        assert w._max_turns_per_tick == 25
        assert w._recent_facts_limit == 5
        assert w._tick_interval_s == pytest.approx(90.0)

    def test_fact_supersede_knobs(self) -> None:
        memory = MemoryProvidersConfig(
            fact_supersede=FactSupersedeConfig(
                batch_size=2, tick_interval_s=120.0, priors_max=6
            )
        )
        [w] = create_projectors(
            names=["fact_supersede"], context=self._ctx_with(memory)
        )
        assert isinstance(w, FactSupersedeWorker)
        assert w._batch_size == 2
        assert w._tick_interval_s == pytest.approx(120.0)
        assert w._priors_max == 6
