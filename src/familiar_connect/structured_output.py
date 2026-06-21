"""Structured-output coercion — parse possibly-fenced LLM JSON, tolerate garbage.

Stub: defined so the contract tests can fail on *behavior*, not import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonResult:
    value: Any = None
    parsed_ok: bool = False


def coerce_json(reply: str) -> JsonResult:
    raise NotImplementedError


def coerce_positive_int_list(raw: Any) -> list[int]:
    raise NotImplementedError


def coerce_str_list(raw: Any) -> list[str]:
    raise NotImplementedError
