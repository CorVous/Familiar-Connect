"""Tests for :func:`familiar_connect.diagnostics.spans.span`."""

from __future__ import annotations

import asyncio
import logging
import re

import pytest

from familiar_connect.diagnostics.spans import span

_ANSI_RE = re.compile(r"\x1b\[\d+m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestSpan:
    @pytest.mark.asyncio
    async def test_emits_ms_log_on_async_function(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        @span("demo")
        async def work() -> int:
            await asyncio.sleep(0.005)
            return 42

        with caplog.at_level(logging.INFO, logger="familiar_connect.diagnostics"):
            result = await work()
        assert result == 42
        messages = [_strip_ansi(r.getMessage()) for r in caplog.records]
        records = [m for m in messages if "span=" in m]
        assert records, "no span log emitted"
        assert "span=demo" in records[-1]
        assert re.search(r"ms=\d+", records[-1])

    def test_emits_ms_log_on_sync_function(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        @span("sync-demo")
        def work() -> str:
            return "ok"

        with caplog.at_level(logging.INFO, logger="familiar_connect.diagnostics"):
            assert work() == "ok"
        messages = [_strip_ansi(r.getMessage()) for r in caplog.records]
        records = [m for m in messages if "span=" in m]
        assert records
        assert "span=sync-demo" in records[-1]

    @pytest.mark.asyncio
    async def test_logs_even_on_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        @span("failing")
        async def work() -> None:
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        with (
            caplog.at_level(logging.INFO, logger="familiar_connect.diagnostics"),
            pytest.raises(RuntimeError),
        ):
            await work()

        messages = [_strip_ansi(r.getMessage()) for r in caplog.records]
        records = [m for m in messages if "span=" in m]
        assert records
        assert "span=failing" in records[-1]
        assert "status=error" in records[-1]
