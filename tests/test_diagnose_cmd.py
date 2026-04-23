"""Tests for the ``familiar-connect diagnose`` CLI subcommand."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest  # noqa: TC002 — runtime use via capsys fixture typing

from familiar_connect.cli import create_parser
from familiar_connect.commands.diagnose import _aggregate, diagnose

if TYPE_CHECKING:
    from pathlib import Path

_SPAN_LINE = "2026-04-22 12:00:00 INFO [span] span={name} ms={ms} status={status}"


def _log(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


class TestAggregate:
    def test_parses_simple_span_lines(self) -> None:
        lines = [
            _SPAN_LINE.format(name="llm", ms=100, status="ok"),
            _SPAN_LINE.format(name="llm", ms=200, status="ok"),
            _SPAN_LINE.format(name="tts", ms=80, status="ok"),
            "junk line",
        ]
        summary = _aggregate(iter(lines))
        assert summary["llm"]["count"] == 2
        assert summary["tts"]["count"] == 1
        assert summary["llm"]["p50"] == 150

    def test_tolerates_ansi_coloured_lines(self) -> None:
        line = (
            "\x1b[37mspan=\x1b[0m\x1b[95mllm\x1b[0m "
            "\x1b[37mms=\x1b[0m\x1b[96m42\x1b[0m "
            "\x1b[37mstatus=\x1b[0m\x1b[32mok\x1b[0m"
        )
        summary = _aggregate(iter([line]))
        assert summary["llm"]["count"] == 1
        assert summary["llm"]["last_ms"] == 42


class TestDiagnoseCLI:
    def test_subcommand_registered(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["diagnose", "somefile"])
        assert args.command == "diagnose"
        assert args.paths == ["somefile"]

    def test_runs_against_a_log_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_path = tmp_path / "bot.log"
        log_path.write_text(
            _log([
                _SPAN_LINE.format(name="llm", ms=50, status="ok"),
                _SPAN_LINE.format(name="llm", ms=150, status="ok"),
                _SPAN_LINE.format(name="tts", ms=20, status="ok"),
            ])
        )
        args = argparse.Namespace(paths=[str(log_path)])
        code = diagnose(args)
        assert code == 0
        captured = capsys.readouterr()
        assert "llm" in captured.out
        assert "tts" in captured.out

    def test_runs_against_multiple_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = tmp_path / "a.log"
        b = tmp_path / "b.log"
        a.write_text(_log([_SPAN_LINE.format(name="llm", ms=10, status="ok")]))
        b.write_text(_log([_SPAN_LINE.format(name="llm", ms=30, status="ok")]))
        args = argparse.Namespace(paths=[str(a), str(b)])
        code = diagnose(args)
        assert code == 0
        out = capsys.readouterr().out
        assert "llm" in out

    def test_empty_log_shows_placeholder(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log = tmp_path / "empty.log"
        log.write_text("nothing here\n")
        args = argparse.Namespace(paths=[str(log)])
        assert diagnose(args) == 0
        out = capsys.readouterr().out
        assert "no spans" in out
