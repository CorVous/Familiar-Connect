"""Tests for the minimal :class:`Familiar` bundle.

Post-demolition the bundle is five fields plus a couple of
dependency-injected clients. Providers / processors / monitor /
memory writer / tracker registry / metrics collector are all gone.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from familiar_connect.familiar import Familiar
from familiar_connect.history.store import HistoryStore
from familiar_connect.subscriptions import SubscriptionRegistry
from tests.conftest import build_fake_llm_clients

if TYPE_CHECKING:
    from pathlib import Path


def _seed_familiar_root(
    tmp_path: Path,
    default_profile_path: Path,
    name: str = "test-familiar",
) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(default_profile_path, root / "character.toml")
    return root


class TestLoadFromDisk:
    def test_returns_minimal_bundle(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        root = _seed_familiar_root(tmp_path, default_profile_path)
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=build_fake_llm_clients(),
            defaults_path=default_profile_path,
        )
        assert familiar.id == root.name
        assert familiar.root == root
        assert familiar.config.display_tz == "UTC"
        assert isinstance(familiar.history_store, HistoryStore)
        assert isinstance(familiar.subscriptions, SubscriptionRegistry)
        assert "main_prose" in familiar.llm_clients

    def test_tts_and_transcriber_default_to_none(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        root = _seed_familiar_root(tmp_path, default_profile_path)
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=build_fake_llm_clients(),
            defaults_path=default_profile_path,
        )
        assert familiar.tts_client is None
        assert familiar.transcriber is None

    def test_history_db_created_on_load(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        root = _seed_familiar_root(tmp_path, default_profile_path)
        Familiar.load_from_disk(
            root,
            llm_clients=build_fake_llm_clients(),
            defaults_path=default_profile_path,
        )
        assert (root / "history.db").exists()
