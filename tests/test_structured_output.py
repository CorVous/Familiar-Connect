"""Tests for the shared structured-output coercion module.

Encodes the degradation contract three existing call sites
(:mod:`familiar_connect.sleep.consolidation`,
:mod:`...sleep.opinion_formation`,
:mod:`...processors.fact_extractor`) duplicate today: strip code
fences, pull the SHAPE-SPECIFIC balanced JSON object/array, ``json.loads``
it, and DEGRADE on any failure rather than raise.

``coerce_json`` takes a required ``expect`` keyword so each call site can
ask for the exact shape it parses today — object-only (consolidation,
opinion-formation),
array-only (fact_extractor), or first-blob (``"any"``). This keeps the
module a faithful superset: a reply containing BOTH shapes must not flip
to the wrong one.
"""

from __future__ import annotations

import json

from familiar_connect.structured_output import (
    coerce_json,
    coerce_positive_int_list,
    coerce_str_list,
)


class TestCoerceJson:
    """``coerce_json`` — fence-tolerant, shape-aware, never-raising parse."""

    def test_fenced_object_parses_with_ok_true(self) -> None:
        payload = {"retire": [{"fact_ids": [1]}], "rewrite": []}
        reply = f"```json\n{json.dumps(payload)}\n```"
        result = coerce_json(reply, expect="object")
        assert result.parsed_ok is True
        assert result.value == payload

    def test_fenced_array_parses_with_ok_true(self) -> None:
        payload = [{"text": "a"}, {"text": "b"}]
        reply = f"```json\n{json.dumps(payload)}\n```"
        result = coerce_json(reply, expect="array")
        assert result.parsed_ok is True
        assert result.value == payload

    def test_not_json_degrades_without_raising(self) -> None:
        # The exact phrase the existing degradation tests use.
        result = coerce_json("not json at all", expect="object")
        assert result.parsed_ok is False
        assert result.value is None

    def test_empty_object_is_parsed_ok_not_fumbled(self) -> None:
        # {} → parsed_ok True (model returned an empty object); the
        # empty-vs-fumbled distinction must hold under every expect.
        for expect in ("object", "any"):
            result = coerce_json("{}", expect=expect)
            assert result.parsed_ok is True
            assert result.value == {}

    def test_garbage_is_fumbled_under_every_expect(self) -> None:
        for expect in ("object", "array", "any"):
            result = coerce_json("not json at all", expect=expect)
            assert result.parsed_ok is False
            assert result.value is None

    def test_object_expect_skips_leading_array(self) -> None:
        # Critic probe: a reply with BOTH shapes, array first. An
        # object-expecting site (consolidation) must get the OBJECT, not
        # the array — else consolidation.reply_parse_failed flips True
        # and the plan is zeroed.
        reply = '["b"] and then {"retire":[{"fact_ids":[1]}],"rewrite":[]}'
        result = coerce_json(reply, expect="object")
        assert result.parsed_ok is True
        assert result.value == {"retire": [{"fact_ids": [1]}], "rewrite": []}

    def test_array_expect_skips_leading_object(self) -> None:
        # Critic probe: both shapes, object first. An array-expecting
        # site (fact_extractor) must get the ARRAY, not the object —
        # else facts are silently zeroed.
        reply = '{"a":1} then [{"text":"f","source_turn_ids":[1]}]'
        result = coerce_json(reply, expect="array")
        assert result.parsed_ok is True
        assert result.value == [{"text": "f", "source_turn_ids": [1]}]

    def test_any_expect_keeps_first_blob_behavior(self) -> None:
        # expect="any" preserves the old whichever-starts-first rule:
        # array starts before object here, so the array wins.
        reply = '["b"] then {"k":1}'
        result = coerce_json(reply, expect="any")
        assert result.parsed_ok is True
        assert result.value == ["b"]


class TestCoercePositiveIntList:
    """Distinct positive ints, bools rejected, order preserved."""

    def test_rejects_bools_dupes_and_non_positive_preserving_order(self) -> None:
        # True == 1 in Python; it must NOT slip through as the int 1.
        assert coerce_positive_int_list([3, 3, True, -1, 0, 7]) == [3, 7]

    def test_malformed_int_strings_drop_without_raising(self) -> None:
        # "--5".lstrip("-") == "5" passes .isdigit() but int("--5")
        # crashes — the never-raise promise must hold; malformed items
        # drop rather than blow up the worker.
        assert coerce_positive_int_list(["--5", "3-", "x", "1.5", 0, -2, 3, 3, 7]) == [
            3,
            7,
        ]


class TestCoerceStrList:
    """Distinct non-empty strings, order preserved."""

    def test_keeps_distinct_non_empty_strings_in_order(self) -> None:
        assert coerce_str_list(["a", "a", "", " ", "b"]) == ["a", "b"]
