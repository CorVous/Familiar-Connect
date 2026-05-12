"""Deepgram streaming transcription backend.

Concrete :class:`Transcriber` over Deepgram's `/v1/listen` WebSocket. See
:mod:`familiar_connect.stt.protocol` for the surface contract; this module
holds the WS lifecycle, replay-on-reconnect buffer, and env-override
factory.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import aiohttp

from familiar_connect import log_style as ls
from familiar_connect.config import DeepgramSTTConfig
from familiar_connect.stt.protocol import TranscriptionEvent, TranscriptionResult

if TYPE_CHECKING:
    from typing import Self

_logger = logging.getLogger(__name__)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"
DEFAULT_LANGUAGE = "en"

DEFAULT_IDLE_FINALIZE_S: float = 0.5
"""Silence gap before forcing Deepgram ``Finalize``.

Discord client-side VAD halts RTP during silence, so Deepgram's
endpointer never sees in-stream silence and holds the final until
next speech burst. After this many idle seconds, pump sends
``{"type":"Finalize"}`` to flush.

Also reused as post-replay cushion in ``_reconnect`` — same silence
window the endpointer expects in normal flow.
"""


# ---------------------------------------------------------------------------
# Deepgram streaming transcriber
# ---------------------------------------------------------------------------


class DeepgramTranscriber:
    """Stream PCM audio to Deepgram, output :class:`TranscriptionResult` via queue."""

    def __init__(
        self: Self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = 48000,
        channels: int = 1,
        diarize: bool = False,
        interim_results: bool = True,
        utterance_end_ms: int = 1500,
        vad_events: bool = False,
        endpointing_ms: int = 500,
        smart_format: bool = True,
        punctuate: bool = True,
        keyterms: tuple[str, ...] = (),
        replay_buffer_s: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        self.diarize = diarize
        self.interim_results = interim_results
        self.utterance_end_ms = utterance_end_ms
        self.vad_events = vad_events
        self.endpointing_ms = endpointing_ms
        self.smart_format = smart_format
        self.punctuate = punctuate
        self.keyterms = keyterms
        self.replay_buffer_s = replay_buffer_s

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        # set by ``stop()`` so the receive loop can distinguish
        # self-initiated close from server/transport closes
        self._shutting_down: bool = False
        # set when receive loop sees a server CLOSE frame; writers
        # short-circuit so audio/KeepAlive don't race the closing
        # transport (avoids `ClientConnectionResetError` and the resulting
        # close_code=1006 misclassification of a clean 1000 close).
        self._closing: bool = False
        # sliding window of recent PCM chunks; replayed to new WS on reconnect
        self._replay_buffer: collections.deque[bytes] = collections.deque()
        self._replay_buffer_bytes: int = 0
        # guards concurrent send_audio / replay-drain so bytes stay ordered
        self._send_lock: asyncio.Lock = asyncio.Lock()

    def build_ws_url(self: Self) -> str:
        """Deepgram WebSocket URL with query params.

        ``interim_results`` / ``utterance_end_ms`` only emitted when
        interims enabled (Deepgram requires ``interim_results=true`` for
        ``utterance_end_ms``). ``endpointing`` always emitted — silence
        gap before finalizing a segment. ``keyterm`` repeated per term
        to bias nova-3 toward jargon / proper nouns.
        """
        params: list[tuple[str, str]] = [
            ("model", self.model),
            ("language", self.language),
            ("sample_rate", str(self.sample_rate)),
            ("channels", str(self.channels)),
            ("encoding", "linear16"),
            ("vad_events", str(self.vad_events).lower()),
            ("endpointing", str(self.endpointing_ms)),
            ("smart_format", str(self.smart_format).lower()),
            ("punctuate", str(self.punctuate).lower()),
        ]
        if self.interim_results:
            params.extend([
                ("interim_results", "true"),
                ("utterance_end_ms", str(self.utterance_end_ms)),
            ])
        if self.diarize:
            params.append(("diarize", "true"))
        params.extend(("keyterm", term) for term in self.keyterms)
        return f"{DEEPGRAM_WS_URL}?{urlencode(params)}"

    def build_headers(self: Self) -> dict[str, str]:
        """Auth headers for Deepgram WebSocket."""
        return {"Authorization": f"Token {self.api_key}"}

    def clone(self: Self) -> DeepgramTranscriber:
        """Create transcriber with same config, independent WS connection."""
        c = DeepgramTranscriber(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            sample_rate=self.sample_rate,
            channels=self.channels,
            diarize=self.diarize,
            interim_results=self.interim_results,
            utterance_end_ms=self.utterance_end_ms,
            vad_events=self.vad_events,
            endpointing_ms=self.endpointing_ms,
            smart_format=self.smart_format,
            punctuate=self.punctuate,
            keyterms=self.keyterms,
            replay_buffer_s=self.replay_buffer_s,
        )
        # carry over any env-tuned class attrs the factory bumped
        c._KEEPALIVE_INTERVAL = self._KEEPALIVE_INTERVAL
        c._MAX_RECONNECTS = self._MAX_RECONNECTS
        c._RECONNECT_BACKOFF_CAP = self._RECONNECT_BACKOFF_CAP
        c._IDLE_CLOSE_S = self._IDLE_CLOSE_S
        return c

    def _parse_response(self: Self, data: dict[str, Any]) -> TranscriptionResult | None:
        """Parse Deepgram response. Returns ``None`` for non-results or empty."""
        if data.get("type") != "Results":
            return None

        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            return None

        best = alternatives[0]
        transcript = best.get("transcript", "")
        if not transcript:
            return None

        # extract speaker from first word if diarization is active
        speaker: int | None = None
        words = best.get("words", [])
        if words and "speaker" in words[0]:
            speaker = int(words[0]["speaker"])

        start = float(data.get("start", 0.0))
        duration = float(data.get("duration", 0.0))

        return TranscriptionResult(
            text=transcript,
            is_final=bool(data.get("is_final")),
            start=start,
            end=start + duration,
            confidence=float(best.get("confidence", 0.0)),
            speaker=speaker,
        )

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def _ws_connect(
        self: Self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
    ) -> aiohttp.ClientWebSocketResponse:
        """Open WS connection. Extracted for testability."""
        return await session.ws_connect(url, headers=headers)

    async def start(self: Self, output: asyncio.Queue[TranscriptionEvent]) -> None:
        """Connect to Deepgram and begin receiving transcription events.

        Output queue carries :class:`TranscriptionResult`s in wire order,
        including interims (``is_final=False``) when ``interim_results=True``.
        """
        url = self.build_ws_url()
        _logger.info(
            f"{ls.tag('🔌 WebSocket', ls.LG)} "
            f"{ls.word('Deepgram', ls.C)} "
            f"{ls.kv('url', url, vc=ls.LW)}"
        )
        self._session = aiohttp.ClientSession()
        self._ws = await self._ws_connect(
            self._session,
            url,
            self.build_headers(),
        )
        _logger.info(
            f"{ls.tag('🔌 WebSocket', ls.G)} "
            f"{ls.word('Deepgram', ls.C)} "
            f"{ls.kv('status', str(self._ws.close_code or 'none'), vc=ls.LW)}"
        )
        self._receive_task = asyncio.create_task(self._receive_loop(output))
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    def _buffer_chunk(self: Self, data: bytes) -> None:
        """Append to sliding replay window, evicting oldest when over budget."""
        self._replay_buffer.append(data)
        self._replay_buffer_bytes += len(data)
        max_bytes = int(self.replay_buffer_s * self.sample_rate * self.channels * 2)
        while self._replay_buffer_bytes > max_bytes and self._replay_buffer:
            evicted = self._replay_buffer.popleft()
            self._replay_buffer_bytes -= len(evicted)

    async def send_audio(self: Self, data: bytes) -> None:
        """Send PCM bytes; buffer for replay on reconnect.

        Chunk lands in replay window before send — mid-send drops still
        get buffered. Raises ``RuntimeError`` if called before :meth:`start`.
        """
        if self._ws is None:
            msg = "Transcriber is not connected — call start() first"
            raise RuntimeError(msg)
        async with self._send_lock:
            self._buffer_chunk(data)
            # `_closing` covers the window between server CLOSE frame and
            # transport `closed=True`; writing in that window raises
            # `ClientConnectionResetError` and corrupts the close handshake.
            if self._ws.closed or self._closing:
                return
            with contextlib.suppress(Exception):
                await self._ws.send_bytes(data)

    async def finalize(self: Self) -> None:
        """Force Deepgram to flush buffered segment as a final.

        Sends ``{"type":"Finalize"}``. Discord client-side VAD drops RTP
        during silence, so without explicit flush the endpointer holds
        the final until next speech burst. No-op if WS closed or never
        started — safe from idle-watchdog paths.
        """
        ws = self._ws
        if ws is None or ws.closed:
            return
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "Finalize"})

    async def stop(self: Self) -> None:
        """Gracefully close Deepgram connection."""
        # flip before CloseStream so a late close frame in the receive
        # loop doesn't race into a reconnect
        self._shutting_down = True
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None

        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.send_json({"type": "CloseStream"})
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._ws = None

        if self._receive_task is not None:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
            self._receive_task = None

        if self._session is not None:
            await self._session.close()
            self._session = None

    _MAX_RECONNECTS: int = 5
    _RECONNECT_DELAY: float = 1.0  # base delay; first attempt is immediate
    _RECONNECT_BACKOFF_CAP: float = 16.0  # max backoff in seconds
    _KEEPALIVE_INTERVAL: float = 3.0
    # bot-side per-user idle window. read by ``bot._start_voice_intake`` to
    # spawn the idle watchdog. 0 disables. lives on the transcriber so the
    # env-var factory and ``clone()`` carry it without bot.py importing
    # config plumbing of its own.
    _IDLE_CLOSE_S: float = 30.0

    # close codes that indicate a permanent, non-recoverable condition
    # (auth, billing, policy). reconnecting would just fail identically.
    # 1008 = policy violation (RFC 6455). 4xxx = Deepgram application-level.
    _NO_RECONNECT_CLOSE_CODES: frozenset[int] = frozenset({1008})

    @classmethod
    def _should_reconnect(cls, close_code: object) -> bool:
        """Classify close code. False → stop; True → reconnect."""
        if not isinstance(close_code, int):
            # transport-level drop (no close frame seen) — retry
            return True
        if close_code in cls._NO_RECONNECT_CLOSE_CODES:
            return False
        # Deepgram 4xxx codes = application-level errors (auth/billing/quota)
        return not (4000 <= close_code < 5000)

    async def _keepalive_loop(self: Self) -> None:
        """Periodic KeepAlive to prevent Deepgram's ~10s idle timeout.

        Re-reads ``_ws`` each tick to follow reconnects transparently.
        """
        while True:
            await asyncio.sleep(self._KEEPALIVE_INTERVAL)
            ws = self._ws
            if ws is None or ws.closed or self._closing:
                continue
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "KeepAlive"})

    async def _reconnect(self: Self) -> None:
        """Close old session, open fresh one; drain replay buffer to new WS."""
        # tear down old connection
        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()

        # open new connection
        self._session = aiohttp.ClientSession()
        url = self.build_ws_url()
        self._ws = await self._ws_connect(
            self._session,
            url,
            self.build_headers(),
        )
        # fresh socket — clear the closing flag set by the prior CLOSE frame
        self._closing = False
        _logger.info(
            f"{ls.tag('🔄 WebSocket', ls.Y)} "
            f"{ls.word('Deepgram', ls.C)} "
            f"{ls.word('reconnected', ls.W)}"
        )

        # replay buffered audio; hold send_lock so send_audio callers
        # queue behind the drain rather than interleaving
        async with self._send_lock:
            chunks_replayed = len(self._replay_buffer)
            replay_bytes = self._replay_buffer_bytes
            for chunk in self._replay_buffer:
                with contextlib.suppress(Exception):
                    await self._ws.send_bytes(chunk)
            self._replay_buffer.clear()
            self._replay_buffer_bytes = 0

        if chunks_replayed:
            # wait ~replay duration before Finalize so Deepgram can process
            # the burst. the replay arrives much faster than real-time, and
            # Finalize emits "what's been transcribed so far" — firing it
            # immediately produces a partial covering only the first few
            # chunks the server had time to process. sleep is outside the
            # send_lock so post-reconnect audio can flow through normally.
            # DEFAULT_IDLE_FINALIZE_S cushion mirrors the silence window the
            # endpointer sees in the normal idle-finalize path.
            replay_s = replay_bytes / (self.sample_rate * self.channels * 2)
            await asyncio.sleep(replay_s + DEFAULT_IDLE_FINALIZE_S)
            async with self._send_lock:
                ws = self._ws
                if ws is not None and not ws.closed:
                    with contextlib.suppress(Exception):
                        await ws.send_json({"type": "Finalize"})
            _logger.info(
                f"{ls.tag('🔁 Replay', ls.C)} "
                f"{ls.kv('chunks', str(chunks_replayed), vc=ls.LW)}"
            )

    async def _receive_loop(
        self: Self,
        output: asyncio.Queue[TranscriptionEvent],
    ) -> None:
        """Read messages from the WebSocket, reconnecting on drops."""
        consecutive_reconnects = 0

        while consecutive_reconnects <= self._MAX_RECONNECTS:
            if self._ws is None:
                return
            # explicit `receive()` instead of `async for` so we observe the
            # CLOSE message itself — aiohttp's `__anext__` swallows CLOSE/
            # CLOSING/CLOSED via `StopAsyncIteration` and only leaves
            # `ws.close_code` behind. seeing CLOSE lets us flip `_closing`
            # immediately so writers (audio pump, KeepAlive) stop racing
            # the closing transport. it also exposes the close-frame reason
            # string in `msg.extra`, which Deepgram occasionally populates.
            close_reason: str | None = None
            while True:
                msg = await self._ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")
                    if msg_type == "Results":
                        result = self._parse_response(data)
                        if result is not None:
                            await output.put(result)
                            # got real data — reset reconnect counter
                            consecutive_reconnects = 0
                    else:
                        # full payload — `Metadata` carries session-end stats
                        # (duration, models, model_info) needed to diagnose
                        # per-user session closes; truncating hides
                        # everything past request_id.
                        _logger.info(
                            f"{ls.tag('Event', ls.C)} "
                            f"{ls.word('Deepgram', ls.C)} "
                            f"{ls.kv('type', msg_type)} "
                            f"{ls.kv('data', msg.data, vc=ls.LW)}"
                        )
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    # server-initiated close frame; freeze writers so
                    # they don't write to the closing transport while
                    # aiohttp finishes the close handshake. CLOSED/ERROR
                    # below are post-close states — `ws.closed` is already
                    # True, so the existing send_audio check handles them.
                    self._closing = True
                    close_reason = msg.extra if isinstance(msg.extra, str) else None
                    break
                elif msg.type in {
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    break

            close_code = self._ws.close_code if self._ws else None
            ws_exc: BaseException | None = None
            if self._ws is not None:
                with contextlib.suppress(Exception):
                    ws_exc = self._ws.exception()
            _logger.info(
                f"{ls.tag('🔌 WebSocket', ls.Y)} "
                f"{ls.word('Deepgram', ls.C)} "
                f"{ls.word('loop-exit', ls.W)} "
                f"{ls.kv('close_code', str(close_code), vc=ls.LW)} "
                f"{ls.kv('reason', repr(close_reason), vc=ls.LW)} "
                f"{ls.kv('exc', repr(ws_exc), vc=ls.LW)}"
            )

            # self-initiated close → exit silently; ``stop()`` handles cleanup
            if self._shutting_down:
                _logger.info(
                    f"{ls.tag('🔌 WebSocket', ls.Y)} "
                    f"{ls.word('Deepgram', ls.C)} "
                    f"{ls.word('shutdown', ls.W)} "
                    f"{ls.kv('close_code', str(close_code), vc=ls.LW)}"
                )
                return

            # non-recoverable close (auth/billing/policy) → stop retrying
            if not self._should_reconnect(close_code):
                _logger.error(
                    f"{ls.tag('🔌 WebSocket', ls.R)} "
                    f"{ls.word('Deepgram', ls.C)} "
                    f"{ls.word('non-recoverable', ls.W)} "
                    f"{ls.kv('close_code', str(close_code), vc=ls.LW)}"
                )
                return

            outage_start = time.monotonic()
            consecutive_reconnects += 1
            # exponential backoff: first attempt is immediate; subsequent
            # failures back off as 1x, 2x, 4x... the base delay up to the cap
            backoff = 0.0
            if consecutive_reconnects > 1:
                exponent = consecutive_reconnects - 2  # attempt 2 → 2^0 = 1
                backoff = min(
                    self._RECONNECT_DELAY * (2**exponent),
                    self._RECONNECT_BACKOFF_CAP,
                )
            attempt = f"{consecutive_reconnects}/{self._MAX_RECONNECTS}"
            _logger.warning(
                f"{ls.tag('🔌 WebSocket', ls.Y)} "
                f"{ls.word('Deepgram', ls.C)} "
                f"{ls.word('reconnecting', ls.W)} "
                f"{ls.kv('close_code', str(close_code), vc=ls.LW)} "
                f"{ls.kv('attempt', attempt, vc=ls.LW)} "
                f"{ls.kv('backoff_s', f'{backoff:.1f}', vc=ls.LW)}"
            )

            try:
                if backoff > 0:
                    await asyncio.sleep(backoff)
                await self._reconnect()
                outage_s = time.monotonic() - outage_start
                _logger.info(
                    f"{ls.tag('🔌 WebSocket', ls.G)} "
                    f"{ls.word('Deepgram', ls.C)} "
                    f"{ls.word('recovered', ls.LG)} "
                    f"{ls.kv('close_code', str(close_code), vc=ls.LW)} "
                    f"{ls.kv('attempt', attempt, vc=ls.LW)} "
                    f"{ls.kv('outage_s', f'{outage_s:.2f}', vc=ls.LW)}"
                )
            except Exception:
                _logger.exception("Deepgram reconnection failed")
                return

        _logger.error(
            f"{ls.tag('🔌 WebSocket', ls.R)} "
            f"{ls.word('Deepgram', ls.C)} "
            f"{ls.word('max reconnects exhausted', ls.W)} "
            f"{ls.kv('attempts', str(self._MAX_RECONNECTS), vc=ls.LW)}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_deepgram_transcriber(
    config: DeepgramSTTConfig | None = None,
) -> DeepgramTranscriber:
    """Build :class:`DeepgramTranscriber` from *config*.

    All non-secret knobs come from ``[providers.stt.deepgram]``;
    ``DEEPGRAM_API_KEY`` is the only env input.

    :raises ValueError: If ``DEEPGRAM_API_KEY`` not set.
    """
    cfg = config or DeepgramSTTConfig()
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        msg = "DEEPGRAM_API_KEY environment variable is required"
        raise ValueError(msg)

    t = DeepgramTranscriber(
        api_key=api_key,
        model=cfg.model,
        language=cfg.language,
        endpointing_ms=cfg.endpointing_ms,
        utterance_end_ms=cfg.utterance_end_ms,
        smart_format=cfg.smart_format,
        punctuate=cfg.punctuate,
        keyterms=cfg.keyterms,
        replay_buffer_s=cfg.replay_buffer_s,
    )
    t._KEEPALIVE_INTERVAL = cfg.keepalive_interval_s  # noqa: SLF001
    t._MAX_RECONNECTS = cfg.reconnect_max_attempts  # noqa: SLF001
    t._RECONNECT_BACKOFF_CAP = cfg.reconnect_backoff_cap_s  # noqa: SLF001
    t._IDLE_CLOSE_S = cfg.idle_close_s  # noqa: SLF001
    return t
