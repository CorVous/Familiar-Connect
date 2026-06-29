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


# Sleep schedule (window/grace) lives in character.toml [sleep] now; the
# catalog entry only marks WHICH activity is window-scheduled. See
# tests/test_config.py::TestLoadCharacterConfig sleep cases.
SLEEP_ENTRY = """\
[[catalog]]
id = "sleep"
label = "asleep"
reachable = false
seed = "The night's dream, told on waking."
"""


class TestPerEntrySchedule:
    """Optional per-entry ``active_days`` / ``active_hours`` (parse only)."""

    def test_both_fields_parsed(self, tmp_path: Path) -> None:
        content = VALID_ENTRY + (
            'active_days = ["mon", "tue", "wed", "thu", "fri"]\n'
            'active_hours = "09:00-17:00"\n'
        )
        path = write_activities(tmp_path, content)
        entry = load_activities_config(path).catalog[0]
        assert entry.active_days == frozenset({0, 1, 2, 3, 4})
        assert entry.active_hours == (time(9, 0), time(17, 0))

    def test_absent_fields_default_to_none(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, VALID_ENTRY)
        entry = load_activities_config(path).catalog[0]
        assert entry.active_days is None
        assert entry.active_hours is None

    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            ('["mon"]', {0}),
            ('["sun"]', {6}),
            ('["mon", "wed", "fri"]', {0, 2, 4}),
        ],
    )
    def test_weekday_mapping_pinned(
        self, tmp_path: Path, tokens: str, expected: set[int]
    ) -> None:
        # Mon=0 .. Sun=6 (datetime.weekday()); the exact mapping is
        # load-bearing for Inc 2/3, so pin individual tokens, not just sets.
        content = VALID_ENTRY + f"active_days = {tokens}\n"
        path = write_activities(tmp_path, content)
        entry = load_activities_config(path).catalog[0]
        assert entry.active_days == frozenset(expected)

    def test_active_hours_midnight_wrap_stored(self, tmp_path: Path) -> None:
        content = VALID_ENTRY + 'active_hours = "22:00-02:30"\n'
        path = write_activities(tmp_path, content)
        entry = load_activities_config(path).catalog[0]
        assert entry.active_hours == (time(22, 0), time(2, 30))

    @pytest.mark.parametrize("token", ['"funday"', '"Mon"'])
    def test_bad_weekday_token_rejected(self, tmp_path: Path, token: str) -> None:
        # Unknown token, and a wrong-case token (tokens are canonical lowercase).
        content = VALID_ENTRY + f"active_days = [{token}]\n"
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError) as excinfo:
            load_activities_config(path)
        assert "creek_walk" in str(excinfo.value)
        assert "active_days" in str(excinfo.value)

    def test_non_string_weekday_token_rejected(self, tmp_path: Path) -> None:
        content = VALID_ENTRY + 'active_days = ["mon", 5]\n'
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError) as excinfo:
            load_activities_config(path)
        assert "creek_walk" in str(excinfo.value)
        assert "active_days" in str(excinfo.value)

    @pytest.mark.parametrize("value", ['"mon"', "5"])
    def test_non_list_active_days_rejected(self, tmp_path: Path, value: str) -> None:
        content = VALID_ENTRY + f"active_days = {value}\n"
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError) as excinfo:
            load_activities_config(path)
        assert "creek_walk" in str(excinfo.value)
        assert "active_days" in str(excinfo.value)

    def test_empty_active_days_rejected(self, tmp_path: Path) -> None:
        content = VALID_ENTRY + "active_days = []\n"
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError) as excinfo:
            load_activities_config(path)
        assert "creek_walk" in str(excinfo.value)
        assert "active_days" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["9am-5pm", "17:00-17:00"])
    def test_malformed_active_hours_rejected(self, tmp_path: Path, value: str) -> None:
        content = VALID_ENTRY + f'active_hours = "{value}"\n'
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError) as excinfo:
            load_activities_config(path)
        # "HH:MM" comes only from parse_hhmm_range's message, not the generic
        # unknown-key error — proves the parse path is actually wired in.
        assert "HH:MM" in str(excinfo.value)
        # Per-entry context: the failing entry id is named.
        assert "creek_walk" in str(excinfo.value)


class TestSleepCatalogEntry:
    """Reserved ``sleep`` catalog id — duration optional (fixed wake)."""

    def test_sleep_entry_parses(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, SLEEP_ENTRY)
        cfg = load_activities_config(path)
        entry = cfg.catalog[0]
        assert entry.id == "sleep"
        assert entry.reachable is False

    def test_duration_minutes_optional_for_sleep_entry(self, tmp_path: Path) -> None:
        path = write_activities(tmp_path, SLEEP_ENTRY)
        cfg = load_activities_config(path)
        assert cfg.catalog[0].duration_minutes is None

    def test_duration_minutes_still_parsed_if_present(self, tmp_path: Path) -> None:
        content = SLEEP_ENTRY + "duration_minutes = [400, 500]\n"
        path = write_activities(tmp_path, content)
        cfg = load_activities_config(path)
        assert cfg.catalog[0].duration_minutes == (400, 500)

    def test_non_sleep_entry_still_requires_duration(self, tmp_path: Path) -> None:
        content = VALID_ENTRY.replace("duration_minutes = [20, 45]\n", "")
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError, match="duration_minutes"):
            load_activities_config(path)

    def test_window_no_longer_a_catalog_key(self, tmp_path: Path) -> None:
        """Window relocated to character.toml — not a valid catalog key."""
        content = SLEEP_ENTRY + 'window = "00:00-08:00"\n'
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError, match="unknown keys"):
            load_activities_config(path)

    def test_grace_no_longer_a_catalog_key(self, tmp_path: Path) -> None:
        content = SLEEP_ENTRY + "grace_minutes = 30\n"
        path = write_activities(tmp_path, content)
        with pytest.raises(ConfigError, match="unknown keys"):
            load_activities_config(path)

    def test_dataclass_has_no_window_or_grace(self) -> None:
        entry = ActivityType(id="x", label="x", duration_minutes=(1, 2), seed="s")
        assert not hasattr(entry, "window")
        assert not hasattr(entry, "grace_minutes")
