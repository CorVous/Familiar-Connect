"""Structural-conformance tests for bus Protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.bus.protocols import (
    BackpressurePolicy,
    Processor,
    StreamSource,
)

if TYPE_CHECKING:
    from familiar_connect.bus.envelope import Event
    from familiar_connect.bus.protocols import EventBus


class _DummySource:
    name = "dummy"

    async def run(self, bus: EventBus) -> None:  # noqa: ARG002
        return


class _DummyProcessor:
    name = "dummy"
    topics: tuple[str, ...] = ("discord.text",)

    async def handle(self, event: Event, bus: EventBus) -> None:  # noqa: ARG002
        return


class _NotASource:
    """Missing ``run`` attribute."""

    name = "broken"


class TestProtocols:
    def test_dummy_source_satisfies_protocol(self) -> None:
        assert isinstance(_DummySource(), StreamSource)

    def test_dummy_processor_satisfies_protocol(self) -> None:
        assert isinstance(_DummyProcessor(), Processor)

    def test_missing_method_rejects_structural_check(self) -> None:
        assert not isinstance(_NotASource(), StreamSource)


class TestBackpressurePolicy:
    def test_all_policies_listed(self) -> None:
        names = {p.name for p in BackpressurePolicy}
        assert names == {"BLOCK", "DROP_OLDEST", "DROP_NEWEST", "UNBOUNDED"}
