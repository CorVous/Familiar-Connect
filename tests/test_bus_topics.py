"""Sanity tests for topic constants."""

from __future__ import annotations

from familiar_connect.bus import topics


def test_topics_are_strings_and_unique() -> None:
    values = [
        getattr(topics, name) for name in dir(topics) if name.startswith("TOPIC_")
    ]
    assert values, "no topics defined"
    assert all(isinstance(v, str) for v in values)
    assert len(values) == len(set(values)), "duplicate topic string"


def test_core_topics_present() -> None:
    assert topics.TOPIC_DISCORD_TEXT == "discord.text"
    assert topics.TOPIC_VOICE_TRANSCRIPT_FINAL == "voice.transcript.final"
    assert topics.TOPIC_TWITCH_EVENT == "twitch.event"
