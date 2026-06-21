"""Tests for the shared structured-output coercion module.

Encodes the degradation contract three existing call sites
(:mod:`familiar_connect.sleep.hygiene`, :mod:`...sleep.dream`,
:mod:`...processors.fact_extractor`) duplicate today: strip code
fences, pull the first balanced JSON object/array, ``json.loads`` it,
and DEGRADE on any failure rather than raise.
"""

from __future__ import annotations

import json

from familiar_connect.structured_output import (
    coerce_json,
    coerce_positive_int_list,
    coerce_str_list,
)


class TestCoerceJson:
    """``coerce_json`` — fence-tolerant, never-raising JSON parse."""

    def test_fenced_object_parses_with_ok_true(self) -> None:
        payload = {"retire": [{"fact_ids": [1]}], "rewrite": []}
        reply = f"```json\n{json.dumps(payload)}\n```"
        result = coerce_json(reply)
        assert result.parsed_ok is True
        assert result.value == payload

    def test_fenced_array_parses_with_ok_true(self) -> None:
        payload = [{"text": "a"}, {"text": "b"}]
        reply = f"```json\n{json.dumps(payload)}\n```"
        result = coerce_json(reply)
        assert result.parsed_ok is True
        assert result.value == payload

    def test_not_json_degrades_without_raising(self) -> None:
        # The exact phrase the existing degradation tests use.
        result = coerce_json("not json at all")
        assert result.parsed_ok is False
        assert result.value is None


class TestCoercePositiveIntList:
    """Distinct positive ints, bools rejected, order preserved."""

    def test_rejects_bools_dupes_and_non_positive_preserving_order(self) -> None:
        # True == 1 in Python; it must NOT slip through as the int 1.
        assert coerce_positive_int_list([3, 3, True, -1, 0, 7]) == [3, 7]


class TestCoerceStrList:
    """Distinct non-empty strings, order preserved."""

    def test_keeps_distinct_non_empty_strings_in_order(self) -> None:
        assert coerce_str_list(["a", "a", "", " ", "b"]) == ["a", "b"]
