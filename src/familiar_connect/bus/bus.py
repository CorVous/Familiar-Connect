"""In-process event bus with per-topic backpressure policies.

Topic-keyed fan-out. Every :meth:`InProcessEventBus.subscribe` call
creates an isolated bounded/unbounded queue; per-subscription
:class:`BackpressurePolicy` decides what happens when the queue is
full. See plan § Design.1.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from enum import Enum
from typing import TYPE_CHECKING

from familiar_connect.bus.protocols import BackpressurePolicy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from familiar_connect.bus.envelope import Event


_logger = logging.getLogger(__name__)

_DEFAULT_MAXSIZE: dict[BackpressurePolicy, int] = {
    BackpressurePolicy.BLOCK: 64,
    BackpressurePolicy.DROP_OLDEST: 64,
    BackpressurePolicy.DROP_NEWEST: 64,
    BackpressurePolicy.UNBOUNDED: 0,
}


class Lifecycle(Enum):
    """Bus lifecycle states. See plan § Design.1 *Bus lifecycle*."""

    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"


class _Subscription:
    """One subscriber's queue + policy. Owned by the bus."""

    def __init__(
        self,
        topics: Iterable[str],
        policy: BackpressurePolicy,
        maxsize: int,
    ) -> None:
        self.topics: frozenset[str] = frozenset(topics)
        self.policy = policy
        effective = maxsize if maxsize > 0 else _DEFAULT_MAXSIZE[policy]
        if policy is BackpressurePolicy.UNBOUNDED:
            self._queue: asyncio.Queue[Event] = asyncio.Queue()
        else:
            self._queue = asyncio.Queue(maxsize=effective)
        self._closed = asyncio.Event()

    async def put(self, event: Event) -> None:
        if self._closed.is_set():
            return
        policy = self.policy
        if policy is BackpressurePolicy.BLOCK or policy is BackpressurePolicy.UNBOUNDED:
            await self._queue.put(event)
            return
        if policy is BackpressurePolicy.DROP_OLDEST:
            while self._queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(event)
            return
        if policy is BackpressurePolicy.DROP_NEWEST:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(event)
            return

    def close(self) -> None:
        self._closed.set()

    async def iterator(self) -> AsyncIterator[Event]:
        """Drain queued events until the subscription is closed.

        Yields:
            Event: next queued event; raises :class:`StopAsyncIteration`
            once the subscription is closed and the queue is drained.

        """
        while True:
            if self._closed.is_set():
                # drain what's already queued, then exit
                while True:
                    try:
                        yield self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
            get_task = asyncio.create_task(self._queue.get())
            closed_task = asyncio.create_task(self._closed.wait())
            try:
                done, pending = await asyncio.wait(
                    (get_task, closed_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                get_task.cancel()
                closed_task.cancel()
                raise
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            if get_task in done and not get_task.cancelled():
                yield get_task.result()
                continue
            # closed — drain and exit
            while True:
                try:
                    yield self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    return


class InProcessEventBus:
    """Topic-keyed pub/sub, in-process only.

    Kept behind :class:`familiar_connect.bus.protocols.EventBus`
    Protocol so a future process-spanning impl can replace it without
    touching processors.
    """

    def __init__(self) -> None:
        self.lifecycle: Lifecycle = Lifecycle.STARTING
        self._subscriptions: list[_Subscription] = []

    async def start(self) -> None:
        if self.lifecycle is not Lifecycle.STARTING:
            return
        self.lifecycle = Lifecycle.RUNNING

    async def shutdown(self) -> None:
        if self.lifecycle is Lifecycle.STOPPED:
            return
        self.lifecycle = Lifecycle.DRAINING
        for sub in self._subscriptions:
            sub.close()
        # give subscribers one loop tick to drain before declaring stopped
        await asyncio.sleep(0)
        self.lifecycle = Lifecycle.STOPPED

    async def publish(self, event: Event) -> None:
        if self.lifecycle is Lifecycle.STOPPED:
            _logger.warning("publish after stop: topic=%s", event.topic)
            return
        for sub in self._subscriptions:
            if event.topic in sub.topics:
                await sub.put(event)

    def subscribe(
        self,
        topics: tuple[str, ...],
        *,
        policy: BackpressurePolicy = BackpressurePolicy.BLOCK,
        maxsize: int = 0,
    ) -> AsyncIterator[Event]:
        sub = _Subscription(topics, policy, maxsize)
        self._subscriptions.append(sub)
        return sub.iterator()
