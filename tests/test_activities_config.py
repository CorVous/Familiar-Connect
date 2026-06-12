"""Tests for activities catalog config loader.

Covers ``activities.toml`` sidecar loading (missing file ⇒ disabled,
lorebook precedent), deep-merge over ``_default``, strict validation
(unknown keys, duration ranges, content_source whitelist), and
``active_hours`` parsing including midnight wrap.
"""

from __future__ import annotations

import dataclasses
from datetime import time
from pathlib import Path

import pytest

from familiar_connect.activities import (
    ActivitiesConfig,
    ActivityType,
    load_activities_config,
)
from familiar_connect.config import ConfigError

VALID_ENTRY = """\
[[catalog]]
id = "creek_walk"
label = "out for a creek walk"
duration_minutes = [20, 45]
reachable = true
content_source = "authored"
seed = "Walk along the creek behind the house."
"""


def write_activities(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "activities.toml"
    path.write_text(content)
    return path


class TestDataclassDefaults:
    def test_config_knob_defaults(self) -> None:
        cfg = ActivitiesConfig()
        assert cfg.catalog == ()
        assert cfg.archive_after_minutes == 45
        assert cfg.idle_nudge_minutes == 20
        assert cfg.min_gap_minutes == 90
        assert cfg.active_hours is None

    def test_empty_catalog_is_disabled_and_falsy(self) -> None:
        cfg = ActivitiesConfig()
        assert cfg.enabled is False
        assert not cfg

    def test_populated_catalog_is_enabled_and_truthy(self) -> None:
        entry = ActivityType(
            id="creek_walk",
            label="out for a creek walk",
            duration_minutes=(20, 45),
            reachable=True,
            content_source="authored",
            seed="Walk along the creek.",
        )
        cfg = ActivitiesConfig(catalog=(entry,))
        assert cfg.enabled is True
        assert cfg

    def test_dataclasses_frozen(self) -> None:
        cfg = ActivitiesConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.archive_after_minutes = 1  # ty: ignore[invalid-assignment]


class TestLoadActivitiesConfig:
    def test_missing_file_yields_disabled_config(self, tmp_path: Path) -> None:
        cfg = load_activities_config(tmp_path / "activities.toml")
        assert isinstance(cfg, ActivitiesConfig)
        assert not cfg
        assert cfg.enabled is False

    def test_knobs_without_catalog_stays_disabled(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, "archive_after_minutes = 60\n")
        cfg = load_activities_config(path)
        assert cfg.archive_after_minutes == 60
        assert not cfg

    def test_valid_entry_parsed(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, VALID_ENTRY)
        cfg = load_activities_config(path)
        assert cfg.enabled is True
        assert len(cfg.catalog) == 1
        entry = cfg.catalog[0]
        assert entry.id == "creek_walk"
        assert entry.label == "out for a creek walk"
        assert entry.duration_minutes == (20, 45)
        assert entry.reachable is True
        assert entry.content_source == "authored"
        assert entry.seed == "Walk along the creek behind the house."

    def test_entry_optional_fields_default(self, tmp_path: Path) -> None:
        path = write_activities(
            tmp_path,
            """\
[[catalog]]
id = "creek_walk"
label = "out for a creek walk"
duration_minutes = [20, 45]
seed = "Walk along the creek."
""",
        )
        cfg = load_activities_config(path)
        entry = cfg.catalog[0]
        assert entry.reachable is True
        assert entry.content_source == "authored"

    def test_malformed_toml_raises(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, "not [valid toml\n")
        with pytest.raises(ConfigError, match="failed to parse TOML"):
            load_activities_config(path)

    def test_merges_over_defaults(self, tmp_path: Path) -> None:
        defaults = tmp_path / "defaults.toml"
        defaults.write_text("archive_after_minutes = 30\nidle_nudge_minutes = 15\n")
        path = write_activities(tmp_path, "archive_after_minutes = 60\n" + VALID_ENTRY)
        cfg = load_activities_config(path, defaults_path=defaults)
        assert cfg.archive_after_minutes == 60  # target overrides default
        assert cfg.idle_nudge_minutes == 15  # default survives
        assert cfg.enabled is True

    def test_missing_defaults_file_ok(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, VALID_ENTRY)
        cfg = load_activities_config(path, defaults_path=tmp_path / "missing.toml")
        assert cfg.enabled is True


class TestKnobValidation:
    @pytest.mark.parametrize(
        "knob",
        ["archive_after_minutes", "idle_nudge_minutes", "min_gap_minutes"],
    )
    def test_knob_must_be_positive_int(self, tmp_path: Path, knob: str) -> None:
        path = write_activities(tmp_path, f"{knob} = 0\n")
        with pytest.raises(ConfigError, match=knob):
            load_activities_config(path)

    def test_knob_rejects_non_int(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, 'min_gap_minutes = "soon"\n')
        with pytest.raises(ConfigError, match="min_gap_minutes"):
            load_activities_config(path)

    def test_unknown_top_level_key_rejected(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, "archive_minutes = 45\n")
        with pytest.raises(ConfigError, match="archive_minutes"):
            load_activities_config(path)


class TestCatalogValidation:
    @pytest.mark.parametrize("missing", ["id", "label", "duration_minutes", "seed"])
    def test_required_keys(self, tmp_path: Path, missing: str) -> None:
        lines = [
            line
            for line in VALID_ENTRY.splitlines()
            if not line.startswith(f"{missing} ")
        ]
        path = write_activities(tmp_path, "\n".join(lines) + "\n")
        with pytest.raises(ConfigError, match=missing):
            load_activities_config(path)

    def test_unknown_entry_key_rejected(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, VALID_ENTRY + 'colour = "blue"\n')
        with pytest.raises(ConfigError, match="colour"):
            load_activities_config(path)

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, VALID_ENTRY + "\n" + VALID_ENTRY)
        with pytest.raises(ConfigError, match="duplicate"):
            load_activities_config(path)

    @pytest.mark.parametrize(
        "value",
        ["[20]", "[20, 45, 60]", "[0, 45]", "[45, 20]", '["a", "b"]', "20", "[-5, 45]"],
    )
    def test_duration_minutes_invalid(self, tmp_path: Path, value: str) -> None:
        content = VALID_ENTRY.replace(
            "duration_minutes = [20, 45]", f"duration_minutes = {value}"
        )
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError, match="duration_minutes"):
            load_activities_config(path)

    def test_duration_minutes_equal_bounds_ok(self, tmp_path: Path) -> None:
        content = VALID_ENTRY.replace("[20, 45]", "[30, 30]")
        path = write_activities(tmp_path, content)
        cfg = load_activities_config(path)
        assert cfg.catalog[0].duration_minutes == (30, 30)

    def test_content_source_adapter_reserved(self, tmp_path: Path) -> None:
        content = VALID_ENTRY.replace('"authored"', '"adapter"')
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError, match="reserved"):
            load_activities_config(path)

    def test_content_source_unknown_rejected(self, tmp_path: Path) -> None:
        content = VALID_ENTRY.replace('"authored"', '"telepathy"')
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError, match="content_source"):
            load_activities_config(path)

    def test_empty_id_rejected(self, tmp_path: Path) -> None:
        content = VALID_ENTRY.replace('id = "creek_walk"', 'id = ""')
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError, match="id"):
            load_activities_config(path)


class TestActiveHours:
    def test_parsed_to_times(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, 'active_hours = "10:00-23:00"\n')
        cfg = load_activities_config(path)
        assert cfg.active_hours == (time(10, 0), time(23, 0))

    def test_midnight_wrap_allowed(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, 'active_hours = "22:00-02:30"\n')
        cfg = load_activities_config(path)
        assert cfg.active_hours == (time(22, 0), time(2, 30))

    @pytest.mark.parametrize(
        "value",
        ["10:00", "10-23", "25:00-23:00", "10:99-23:00", "always", ""],
    )
    def test_bad_format_rejected(self, tmp_path: Path, value: str) -> None:
        path = write_activities(tmp_path, f'active_hours = "{value}"\n')
        with pytest.raises(ConfigError, match="active_hours"):
            load_activities_config(path)

    def test_equal_start_end_rejected(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, 'active_hours = "10:00-10:00"\n')
        with pytest.raises(ConfigError, match="active_hours"):
            load_activities_config(path)


class TestDefaultSkeleton:
    def test_ships_disabled(self) -> None:
        skeleton = Path("data/familiars/_default/activities.toml")
        assert skeleton.exists()
        cfg = load_activities_config(skeleton)
        assert not cfg
