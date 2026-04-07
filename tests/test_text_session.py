"""Tests for the TextSession registry."""

import pytest

from familiar_connect.llm import Message
from familiar_connect.text_session import (
    SessionError,
    TextSession,
    clear_session,
    get_session,
    set_session,
)


@pytest.fixture(autouse=True)
def _reset_session():
    """Ensure each test starts with no active session.

    Yields:
        None

    """
    clear_session()
    yield
    clear_session()


class TestTextSession:
    def test_dataclass_fields(self) -> None:
        session = TextSession(channel_id=42, system_prompt="You are X.")
        assert session.channel_id == 42
        assert session.system_prompt == "You are X."
        assert session.history == []

    def test_history_is_independent_per_instance(self) -> None:
        a = TextSession(channel_id=1, system_prompt="")
        b = TextSession(channel_id=2, system_prompt="")
        a.history.append(Message(role="user", content="hi"))
        assert b.history == []


class TestSessionRegistry:
    def test_initially_empty(self) -> None:
        assert get_session() is None

    def test_set_and_get(self) -> None:
        session = TextSession(channel_id=99, system_prompt="test")
        set_session(session)
        assert get_session() is session

    def test_clear_removes_session(self) -> None:
        set_session(TextSession(channel_id=1, system_prompt=""))
        clear_session()
        assert get_session() is None

    def test_set_when_occupied_raises(self) -> None:
        set_session(TextSession(channel_id=1, system_prompt=""))
        with pytest.raises(SessionError):
            set_session(TextSession(channel_id=2, system_prompt=""))

    def test_clear_when_empty_is_a_noop(self) -> None:
        clear_session()  # should not raise
        assert get_session() is None
