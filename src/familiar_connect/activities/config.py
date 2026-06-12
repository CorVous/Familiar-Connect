"""Activities catalog + knobs from ``activities.toml``.

Sidecar per familiar: ``data/familiars/<id>/activities.toml``
(lorebook precedent). Missing file or empty catalog ⇒ disabled
config (falsy) — engine never constructed. Present-but-invalid
content fails loudly with :class:`ConfigError` (character.toml
precedent), so typos don't silently disable a knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import TYPE_CHECKING, cast

from familiar_connect.config import ConfigError, _deep_merge, _read_toml

if TYPE_CHECKING:
    from pathlib import Path


CONTENT_SOURCES: frozenset[str] = frozenset({"authored"})
"""Allowed ``content_source`` values.

``"adapter"`` reserved for future adapter-backed types (e.g.
youtube) — rejected with explicit message until implemented.
"""

SLEEP_TYPE_ID = "sleep"
"""Reserved catalog id — window-scheduled sleep activity.

Only this entry may carry ``window``/``grace_minutes``; with a
``window``, ``duration_minutes`` is optional (return is fixed at
window end, never a rolled duration).
"""

_ENTRY_REQUIRED: frozenset[str] = frozenset({"id", "label", "duration_minutes", "seed"})
_ENTRY_KEYS: frozenset[str] = _ENTRY_REQUIRED | {
    "reachable",
    "content_source",
    "window",
    "grace_minutes",
}
_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "archive_after_minutes",
    "idle_nudge_minutes",
    "min_gap_minutes",
    "active_hours",
    "catalog",
})


@dataclass(frozen=True)
class ActivityType:
    """One ``[[catalog]]`` row from ``activities.toml``.

    :param id: stable identifier referenced by ``start_activity`` tool.
    :param label: presence/status text shown while out.
    :param duration_minutes: ``(lo, hi)`` roll range, ``0 < lo <= hi``;
        ``None`` only on the sleep entry (window-scheduled).
    :param reachable: real @ping while out triggers judgment turn.
    :param content_source: experience text origin; see
        :data:`CONTENT_SOURCES`.
    :param seed: authored prompt seed for experience generation
        (dream prose for the sleep entry).
    :param window: sleep window ``(start, end)`` in display_tz; may
        wrap midnight; :data:`SLEEP_TYPE_ID` entry only.
    :param grace_minutes: minutes after window start before
        force-sleep; sleep entry only.
    """

    id: str
    label: str
    duration_minutes: tuple[int, int] | None = None
    reachable: bool = True
    content_source: str = "authored"
    seed: str = ""
    window: tuple[time, time] | None = None
    grace_minutes: int = 30


@dataclass(frozen=True)
class ActivitiesConfig:
    """Parsed ``activities.toml``: catalog + engine knobs.

    Falsy when catalog empty — callers skip engine construction.

    :param active_hours: ``(start, end)`` in display_tz; may wrap
        midnight; ``None`` = always.
    """

    catalog: tuple[ActivityType, ...] = ()
    archive_after_minutes: int = 45
    idle_nudge_minutes: int = 20
    min_gap_minutes: int = 90
    active_hours: tuple[time, time] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.catalog)

    def __bool__(self) -> bool:
        return self.enabled


def load_activities_config(
    path: Path,
    *,
    defaults_path: Path | None = None,
) -> ActivitiesConfig:
    """Load :class:`ActivitiesConfig` from *path*.

    Deep-merges over *defaults_path* when given; unlike
    ``character.toml``, defaults file is optional (``_default``
    skeleton ships fully commented out). Missing *path* with no
    defaults content ⇒ disabled config.

    :raises ConfigError: invalid TOML, unknown keys, or validation
        failure.
    """
    defaults_data: dict = {}
    if defaults_path is not None:
        defaults_data = _read_toml(defaults_path) or {}
    target_data = _read_toml(path) or {}
    merged = _deep_merge(defaults_data, target_data)
    return _parse_activities_config(merged)


def _parse_activities_config(data: dict) -> ActivitiesConfig:
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        valid = ", ".join(sorted(_TOP_LEVEL_KEYS))
        msg = f"unknown activities.toml keys: {sorted(unknown)}; valid keys: {valid}"
        raise ConfigError(msg)

    knobs: dict[str, int] = {}
    for knob in ("archive_after_minutes", "idle_nudge_minutes", "min_gap_minutes"):
        if knob not in data:
            continue
        value = data[knob]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            msg = f"{knob} must be a positive integer, got {value!r}"
            raise ConfigError(msg)
        knobs[knob] = value

    active_hours: tuple[time, time] | None = None
    if "active_hours" in data:
        active_hours = _parse_active_hours(data["active_hours"])

    catalog_raw = data.get("catalog", [])
    if not isinstance(catalog_raw, list):
        msg = (
            f"[[catalog]] must be an array of tables, got {type(catalog_raw).__name__}"
        )
        raise ConfigError(msg)
    catalog = tuple(
        _parse_catalog_entry(raw, idx) for idx, raw in enumerate(catalog_raw)
    )

    seen: set[str] = set()
    for entry in catalog:
        if entry.id in seen:
            msg = f"duplicate catalog id {entry.id!r}"
            raise ConfigError(msg)
        seen.add(entry.id)

    return ActivitiesConfig(catalog=catalog, active_hours=active_hours, **knobs)


def _parse_catalog_entry(raw: object, idx: int) -> ActivityType:
    if not isinstance(raw, dict):
        msg = f"[[catalog]] entry #{idx} must be a table, got {type(raw).__name__}"
        raise ConfigError(msg)
    entry = cast("dict[str, object]", raw)

    unknown = set(entry) - _ENTRY_KEYS
    if unknown:
        valid = ", ".join(sorted(_ENTRY_KEYS))
        msg = (
            f"unknown keys in [[catalog]] entry #{idx}: {sorted(unknown)}; "
            f"valid keys: {valid}"
        )
        raise ConfigError(msg)
    missing = _ENTRY_REQUIRED - set(entry)
    if "window" in entry or entry.get("id") == SLEEP_TYPE_ID:
        # window-scheduled: return fixed at window end, duration unused
        missing -= {"duration_minutes"}
    if missing:
        msg = f"[[catalog]] entry #{idx} missing required keys: {sorted(missing)}"
        raise ConfigError(msg)

    strings: dict[str, str] = {}
    for key in ("id", "label", "seed"):
        value = entry[key]
        if not isinstance(value, str) or not value.strip():
            msg = f"[[catalog]] entry #{idx}: {key} must be a non-empty string"
            raise ConfigError(msg)
        strings[key] = value
    entry_id = strings["id"]

    # window/grace_minutes reserved for the sleep entry — keeps the
    # schedule semantics in one place instead of half-supported
    # windows on arbitrary types
    if entry_id != SLEEP_TYPE_ID:
        for key in ("window", "grace_minutes"):
            if key in entry:
                msg = (
                    f"[[catalog]] {entry_id!r}: {key} is valid only on the "
                    f"reserved {SLEEP_TYPE_ID!r} entry"
                )
                raise ConfigError(msg)
    elif "window" not in entry:
        msg = f"[[catalog]] {SLEEP_TYPE_ID!r}: 'window' is required"
        raise ConfigError(msg)

    window: tuple[time, time] | None = None
    if "window" in entry:
        window = _parse_hhmm_range(entry["window"], key="window")

    grace_minutes = entry.get("grace_minutes", 30)
    if (
        not isinstance(grace_minutes, int)
        or isinstance(grace_minutes, bool)
        or grace_minutes <= 0
    ):
        msg = (
            f"[[catalog]] {entry_id!r}: grace_minutes must be a positive "
            f"integer, got {grace_minutes!r}"
        )
        raise ConfigError(msg)

    duration: tuple[int, int] | None = None
    if "duration_minutes" in entry:
        duration = _parse_duration_minutes(entry["duration_minutes"], entry_id)

    reachable = entry.get("reachable", True)
    if not isinstance(reachable, bool):
        msg = f"[[catalog]] {entry_id!r}: reachable must be a boolean"
        raise ConfigError(msg)

    content_source = entry.get("content_source", "authored")
    if not isinstance(content_source, str):
        msg = f"[[catalog]] {entry_id!r}: content_source must be a string"
        raise ConfigError(msg)
    if content_source == "adapter":
        msg = (
            f"[[catalog]] {entry_id!r}: content_source 'adapter' is "
            "reserved for future adapter-backed types; use 'authored'"
        )
        raise ConfigError(msg)
    if content_source not in CONTENT_SOURCES:
        valid = ", ".join(sorted(CONTENT_SOURCES))
        msg = (
            f"[[catalog]] {entry_id!r}: unknown content_source "
            f"{content_source!r}; valid values: {valid}"
        )
        raise ConfigError(msg)

    return ActivityType(
        id=entry_id,
        label=strings["label"],
        duration_minutes=duration,
        reachable=reachable,
        content_source=content_source,
        seed=strings["seed"],
        window=window,
        grace_minutes=grace_minutes,
    )


def _parse_duration_minutes(value: object, entry_id: str) -> tuple[int, int]:
    msg = (
        f"[[catalog]] {entry_id!r}: duration_minutes must be a "
        f"[lo, hi] pair of minutes with 0 < lo <= hi, got {value!r}"
    )
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(msg)
    lo, hi = value
    if (
        not isinstance(lo, int)
        or isinstance(lo, bool)
        or not isinstance(hi, int)
        or isinstance(hi, bool)
    ):
        raise ConfigError(msg)
    if not 0 < lo <= hi:
        raise ConfigError(msg)
    return (lo, hi)


def _parse_active_hours(value: object) -> tuple[time, time]:
    return _parse_hhmm_range(value, key="active_hours")


def _parse_hhmm_range(value: object, *, key: str) -> tuple[time, time]:
    msg = (
        f"{key} must be 'HH:MM-HH:MM' (may wrap midnight, start != end), got {value!r}"
    )
    if not isinstance(value, str):
        raise ConfigError(msg)
    parts = value.split("-")
    if len(parts) != 2:
        raise ConfigError(msg)
    parsed: list[time] = []
    for part in parts:
        pieces = part.split(":")
        if len(pieces) != 2 or not all(p.isdigit() and len(p) == 2 for p in pieces):
            raise ConfigError(msg)
        hour, minute = int(pieces[0]), int(pieces[1])
        if hour > 23 or minute > 59:
            raise ConfigError(msg)
        parsed.append(time(hour, minute))
    start, end = parsed
    if start == end:
        raise ConfigError(msg)
    return (start, end)
