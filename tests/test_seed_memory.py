"""Tests for authored memory seeding (seed_turns.toml → turns + facts).

Covers: TOML parsing/validation, insertion with provenance,
idempotent re-run, FTS retrievability via the same search path
``RagContextLayer`` uses, embedding drain, watermark discipline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from familiar_connect.commands.seed_memory import (
    SeedEntry,
    SeedFact,
    SeedFile,
    load_seed_file,
    seed_memory,
)
from familiar_connect.embedding import HashEmbedder
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore

FAM = "sapphire"
SAPPHIRE_SEED = (
    Path(__file__).parent.parent / "data" / "familiars" / "sapphire" / "seed_turns.toml"
)
JOURNAL_CH = 999000111222333444

SEED_TOML = """\
channel_id = 999000111222333444

[[entries]]
id = "monksnail-grip"
turn = "The monksnail shifted its grip again. Fourth occasion. The file grows."
facts = [
    { text = "Sapphire thinks the monksnail is a parasite.", importance = 8 },
    { text = "Sapphire has logged four monksnail grip shifts.", importance = 5 },
]

[[entries]]
id = "cor-rest"
turn = "Cor streamed again without proper rest. I have noted the interval."
facts = [
    { text = "Sapphire quietly tracks Cor's rest and wellbeing.", importance = 9 },
]
"""


def _write_seed(tmp_path: Path, content: str = SEED_TOML) -> Path:
    p = tmp_path / "seed_turns.toml"
    p.write_text(content)
    return p


def _seed_file() -> SeedFile:
    return SeedFile(
        channel_id=JOURNAL_CH,
        entries=(
            SeedEntry(
                id="monksnail-grip",
                turn="The monksnail shifted its grip again today.",
                facts=(
                    SeedFact(
                        text=(
                            "Sapphire is convinced the monksnail is a "
                            "mind-controlling parasite."
                        ),
                        importance=8,
                    ),
                ),
            ),
            SeedEntry(
                id="cor-rest",
                turn="Cor streamed again without proper rest.",
                facts=(
                    SeedFact(
                        text="Sapphire quietly tracks Cor's rest and wellbeing.",
                        importance=9,
                    ),
                ),
            ),
        ),
    )


class TestLoadSeedFile:
    def test_parses_entries_and_facts(self, tmp_path: Path) -> None:
        seed = load_seed_file(_write_seed(tmp_path))
        assert seed.channel_id == JOURNAL_CH
        assert len(seed.entries) == 2
        first = seed.entries[0]
        assert first.id == "monksnail-grip"
        assert "monksnail" in first.turn
        assert len(first.facts) == 2
        assert first.facts[0].importance == 8
        assert seed.entries[1].facts[0].importance == 9

    def test_rejects_duplicate_entry_ids(self, tmp_path: Path) -> None:
        dup = SEED_TOML.replace('id = "cor-rest"', 'id = "monksnail-grip"')
        with pytest.raises(ValueError, match="duplicate"):
            load_seed_file(_write_seed(tmp_path, dup))

    def test_rejects_missing_channel_id(self, tmp_path: Path) -> None:
        no_ch = SEED_TOML.replace("channel_id = 999000111222333444\n", "")
        with pytest.raises(ValueError, match="channel_id"):
            load_seed_file(_write_seed(tmp_path, no_ch))

    def test_rejects_entry_without_facts(self, tmp_path: Path) -> None:
        no_facts = (
            SEED_TOML + '\n[[entries]]\nid = "empty"\nturn = "musing"\nfacts = []\n'
        )
        with pytest.raises(ValueError, match="facts"):
            load_seed_file(_write_seed(tmp_path, no_facts))


class TestSeedMemory:
    @pytest.mark.asyncio
    async def test_inserts_turns_and_facts_with_provenance(self) -> None:
        store = HistoryStore(":memory:")
        astore = AsyncHistoryStore(store)
        report = await seed_memory(store=astore, familiar_id=FAM, seed=_seed_file())

        assert report.inserted_turns == 2
        assert report.skipped_turns == 0
        assert report.inserted_facts == 2

        turn = store.lookup_turn_by_platform_message_id(
            familiar_id=FAM, platform_message_id="seed:monksnail-grip"
        )
        assert turn is not None
        assert turn.role == "assistant"
        assert turn.channel_id == JOURNAL_CH

        facts = store.recent_facts(familiar_id=FAM, limit=10)
        assert len(facts) == 2
        monksnail_fact = next(f for f in facts if "monksnail" in f.text)
        assert monksnail_fact.source_turn_ids == (turn.id,)
        assert monksnail_fact.importance == 8
        # global scope — feelings not channel-bound
        assert monksnail_fact.channel_id is None

    @pytest.mark.asyncio
    async def test_rerun_is_idempotent(self) -> None:
        store = HistoryStore(":memory:")
        astore = AsyncHistoryStore(store)
        await seed_memory(store=astore, familiar_id=FAM, seed=_seed_file())
        report = await seed_memory(store=astore, familiar_id=FAM, seed=_seed_file())

        assert report.inserted_turns == 0
        assert report.skipped_turns == 2
        assert report.inserted_facts == 0
        assert len(store.recent_facts(familiar_id=FAM, limit=10)) == 2

    @pytest.mark.asyncio
    async def test_new_entry_added_on_rerun(self) -> None:
        store = HistoryStore(":memory:")
        astore = AsyncHistoryStore(store)
        seed = _seed_file()
        await seed_memory(store=astore, familiar_id=FAM, seed=seed)

        extra = SeedEntry(
            id="cassidy-watch",
            turn="Cassidy seemed quieter than usual tonight.",
            facts=(
                SeedFact(text="Sapphire watches Cassidy's wellbeing.", importance=7),
            ),
        )
        grown = SeedFile(channel_id=seed.channel_id, entries=(*seed.entries, extra))
        report = await seed_memory(store=astore, familiar_id=FAM, seed=grown)

        assert report.inserted_turns == 1
        assert report.skipped_turns == 2
        assert report.inserted_facts == 1

    @pytest.mark.asyncio
    async def test_seeded_facts_searchable_via_fts(self) -> None:
        store = HistoryStore(":memory:")
        astore = AsyncHistoryStore(store)
        await seed_memory(store=astore, familiar_id=FAM, seed=_seed_file())

        hits = store.search_facts(familiar_id=FAM, query="monksnail parasite", limit=5)
        assert hits
        assert any("parasite" in f.text for f in hits)

    @pytest.mark.asyncio
    async def test_embeds_facts_when_embedder_given(self) -> None:
        store = HistoryStore(":memory:")
        astore = AsyncHistoryStore(store)
        embedder = HashEmbedder(dim=8)
        report = await seed_memory(
            store=astore, familiar_id=FAM, seed=_seed_file(), embedder=embedder
        )

        assert report.embedded_facts == 2
        assert (
            store.unembedded_facts(familiar_id=FAM, model=embedder.name, limit=10) == []
        )

    @pytest.mark.asyncio
    async def test_watermark_advanced_on_clean_store(self) -> None:
        store = HistoryStore(":memory:")
        astore = AsyncHistoryStore(store)
        report = await seed_memory(store=astore, familiar_id=FAM, seed=_seed_file())

        assert report.watermark_advanced is True
        # extractor must see nothing pending after seeding
        assert store.turns_since_watermark(familiar_id=FAM, limit=10) == []

    @pytest.mark.asyncio
    async def test_watermark_untouched_when_backlog_exists(self) -> None:
        store = HistoryStore(":memory:")
        astore = AsyncHistoryStore(store)
        store.append_turn(
            familiar_id=FAM,
            channel_id=1,
            role="user",
            content="organic turn the extractor has not processed",
            author=None,
        )
        report = await seed_memory(store=astore, familiar_id=FAM, seed=_seed_file())

        assert report.watermark_advanced is False
        # backlog turn still pending, seed turns also visible to extractor
        pending = store.turns_since_watermark(familiar_id=FAM, limit=10)
        assert len(pending) == 3


# realistic competing facts — retrieval must rank stances above these
_DISTRACTORS = [
    "Cor died to gravity again in the canyon zone.",
    "Cor is planning a horror game stream for Friday.",
    "KaillaDame enjoys puzzle games and dislikes spoilers.",
    "Postbirb has been grinding the same dungeon for a week.",
    "Cor's stream hit a two-year anniversary in spring.",
    "Tarvis usually lurks and rarely posts more than one word.",
    "DawnRaider99 watches every stream from start to finish.",
    "Cor switched to a new microphone after the old one crackled.",
    "Chat voted for the desert route in the community poll.",
    "Vaelith asked about the eastern ward during the exploration arc.",
    "Cor keeps losing duels in PvP with grenade launchers.",
    "Carvel joined the server during the second charity stream.",
    "The community plays board games on Sunday evenings.",
    "Cor's favorite boss fight took eleven attempts on stream.",
    "Luneth draws fan art of stream moments.",
]

# cue (verbatim chat message) → substring identifying the stance fact
# that must rank in top-K, mirroring RagContextLayer's raw-cue query
_CUE_CASES = [
    ("is the monksnail actually real or is it a bit", "mind-controlling parasite"),
    ("Sapphire do you actually care about Cor", "care only privately"),
    ("how is Cassidy doing", "Cassidy's wellbeing"),
    ("what happened in the woods? you never talk about it", "trouble in the woods"),
    ("does Cor ever rest between streams", "rest and overwork"),
    ("why are you always roasting Cor if you love her", "roasting is affection"),
    ("that new guy is all charm isn't he", "distrusts charm"),
    ("remember when Cor called you her familiar", "called her a 'familiar'"),
    ("some stranger claims he knows Cor from way back", "claims about backstory"),
]

_TOP_K = 5


@pytest.fixture(scope="module")
def seeded_store() -> HistoryStore:
    store = HistoryStore(":memory:")
    astore = AsyncHistoryStore(store)
    seed = load_seed_file(SAPPHIRE_SEED)
    asyncio.run(seed_memory(store=astore, familiar_id=FAM, seed=seed))
    for i, text in enumerate(_DISTRACTORS):
        store.append_fact(
            familiar_id=FAM,
            channel_id=1,
            text=text,
            source_turn_ids=[1000 + i],
            importance=5,
        )
    return store


class TestSeedCorpusRetrieval:
    """Smoke test: shipped seed corpus surfaces for canonical chat cues.

    Mirrors production path — raw message as BM25 query via
    ``search_facts`` (same call ``RagContextLayer.build`` makes),
    with distractor facts competing for rank.
    """

    @pytest.mark.parametrize(("cue", "expected_substring"), _CUE_CASES)
    def test_cue_surfaces_stance_in_top_k(
        self, seeded_store: HistoryStore, cue: str, expected_substring: str
    ) -> None:
        hits = seeded_store.search_facts(familiar_id=FAM, query=cue, limit=_TOP_K)
        texts = [f.text for f in hits]
        assert any(expected_substring in t for t in texts), (
            f"cue {cue!r}: expected fact containing {expected_substring!r} "
            f"in top {_TOP_K}, got: {texts}"
        )
