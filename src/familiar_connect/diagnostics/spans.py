"""``@span`` decorator — structured timing logs.

Logs one line per wrapped call via a dedicated logger. Format uses
the project's :mod:`log_style` KV pairs; ``grep span=<name>`` is the
Phase-1 aggregator.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import TYPE_CHECKING, Any, cast

from familiar_connect import log_style as ls

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = logging.getLogger("familiar_connect.diagnostics")


def span(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a function; log ``span=<name> ms=<elapsed>`` on return.

    Works on both sync and async callables. Emits one INFO line per
    wrapped call via the ``familiar_connect.diagnostics`` logger.
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: object, **kwargs: object) -> object:
                t0 = time.perf_counter()
                status = "ok"
                try:
                    return await fn(*args, **kwargs)
                except BaseException:
                    status = "error"
                    raise
                finally:
                    _emit(name, t0, status)

            return cast("Callable[..., Any]", async_wrapper)

        @functools.wraps(fn)
        def sync_wrapper(*args: object, **kwargs: object) -> object:
            t0 = time.perf_counter()
            status = "ok"
            try:
                return fn(*args, **kwargs)
            except BaseException:
                status = "error"
                raise
            finally:
                _emit(name, t0, status)

        return cast("Callable[..., Any]", sync_wrapper)

    return deco


def _emit(name: str, t0: float, status: str) -> None:
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _logger.info(
        f"{ls.tag('span', ls.LM)} "
        f"{ls.kv('span', name, vc=ls.LM)} "
        f"{ls.kv('ms', str(elapsed_ms), vc=ls.LC)} "
        f"{ls.kv('status', status, vc=ls.LG if status == 'ok' else ls.R)}"
    )
