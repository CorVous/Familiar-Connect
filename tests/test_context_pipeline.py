"""Red-first tests for the empty ContextPipeline and its protocols.

Covers familiar_connect.context.pipeline.ContextPipeline and the
protocols in familiar_connect.context.protocols — neither of which
exists yet.

The pipeline orchestrates registered pre-processors and providers
under a scoped ``asyncio.TaskGroup`` with a per-provider deadline,
then hands the collected Contributions to the Budgeter. Individual
provider failures (exceptions or deadline misses) are recorded but
do not poison the rest of the pipeline.

This commit does not exercise the main LLM call or post-processors —
those wait on later roadmap steps in docs/architecture/context-pipeline.md.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from familiar_connect.context.pipeline import (
    ContextPipeline,
    PipelineOutput,
)
from familiar_connect.context.protocols import (
    ContextProvider,
    PostProcessor,
    PreProcessor,
    PreProcessorError,
)
from familiar_connect.context.types import (
    ContextRequest,
    Contribution,
    Layer,
    Modality,
)
from familiar_connect.identity import Author

_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")

# ---------------------------------------------------------------------------
# Test helpers — stub providers and pre-processors
# ---------------------------------------------------------------------------


def _make_request(**overrides: object) -> ContextRequest:
    defaults: dict[str, Any] = {
        "familiar_id": "aria",
        "channel_id": 100,
        "guild_id": 1,
        "author": _ALICE,
        "utterance": "hello",
        "modality": Modality.text,
        "budget_tokens": 2048,
        "deadline_s": 5.0,
    }
    defaults.update(overrides)
    return ContextRequest(**defaults)  # type: ignore[arg-type]


class _StubProvider:
    """A simple provider that returns a fixed set of contributions."""

    def __init__(
        self,
        provider_id: str,
        contributions: list[Contribution],
        *,
        deadline_s: float = 1.0,
        sleep_s: float = 0.0,
    ) -> None:
        self.id = provider_id
        self.deadline_s = deadline_s
        self._contributions = contributions
        self._sleep_s = sleep_s
        self.call_count = 0
        self.last_request: ContextRequest | None = None

    async def contribute(self, request: ContextRequest) -> list[Contribution]:
        self.call_count += 1
        self.last_request = request
        if self._sleep_s > 0:
            await asyncio.sleep(self._sleep_s)
        return list(self._contributions)


class _FailingProvider:
    def __init__(self, provider_id: str, exc: Exception) -> None:
        self.id = provider_id
        self.deadline_s = 1.0
        self._exc = exc

    async def contribute(
        self,
        request: ContextRequest,  # noqa: ARG002
    ) -> list[Contribution]:
        raise self._exc


class _StubPreProcessor:
    """Pre-processor that overwrites the request's utterance."""

    def __init__(self, processor_id: str, new_utterance: str) -> None:
        self.id = processor_id
        self._new_utterance = new_utterance

    async def process(self, request: ContextRequest) -> ContextRequest:
        return replace(request, utterance=self._new_utterance)


class _ContributingPreProcessor:
    """Pre-processor that stashes a Contribution on preprocessor_contributions."""

    def __init__(self, processor_id: str, contribution: Contribution) -> None:
        self.id = processor_id
        self._contribution = contribution

    async def process(self, request: ContextRequest) -> ContextRequest:
        return replace(
            request,
            preprocessor_contributions=(
                *request.preprocessor_contributions,
                self._contribution,
            ),
        )


class _FailingPreProcessor:
    """Pre-processor whose ``process`` raises a configurable exception."""

    def __init__(self, processor_id: str, exc: Exception) -> None:
        self.id = processor_id
        self._exc = exc

    async def process(
        self,
        request: ContextRequest,  # noqa: ARG002
    ) -> ContextRequest:
        raise self._exc


class _RecordingPostProcessor:
    """Post-processor that records every reply it sees and appends a tag."""

    def __init__(self, processor_id: str, tag: str) -> None:
        self.id = processor_id
        self._tag = tag
        self.seen: list[str] = []

    async def process(self, reply_text: str, request: ContextRequest) -> str:  # noqa: ARG002
        self.seen.append(reply_text)
        return f"{reply_text}|{self._tag}"


class _FailingPostProcessor:
    def __init__(self, processor_id: str, exc: Exception) -> None:
        self.id = processor_id
        self._exc = exc

    async def process(
        self,
        reply_text: str,  # noqa: ARG002
        request: ContextRequest,  # noqa: ARG002
    ) -> str:
        raise self._exc


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolsAreRuntimeCheckable:
    def test_stub_provider_is_context_provider(self) -> None:
        stub = _StubProvider("stub", [])
        assert isinstance(stub, ContextProvider)

    def test_stub_pre_processor_is_pre_processor(self) -> None:
        stub = _StubPreProcessor("pre", "hi")
        assert isinstance(stub, PreProcessor)

    def test_post_processor_protocol_exists(self) -> None:
        """PostProcessor is defined for later roadmap steps.

        The first-pass pipeline doesn't invoke it, but its shape has to
        be stable now so later work can bolt against it.
        """
        assert PostProcessor is not None


# ---------------------------------------------------------------------------
# Empty pipeline
# ---------------------------------------------------------------------------


class TestEmptyPipeline:
    @pytest.mark.asyncio
    async def test_empty_pipeline_returns_empty_output(self) -> None:
        pipeline = ContextPipeline(providers=[])
        req = _make_request()
        result = await pipeline.assemble(req, budget_by_layer={})

        assert isinstance(result, PipelineOutput)
        assert result.request == req
        assert result.outcomes == []
        assert result.budget.by_layer == {}
        assert result.budget.dropped == []


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class TestProviderFanOut:
    @pytest.mark.asyncio
    async def test_single_provider_contributions_reach_result(self) -> None:
        contribution = Contribution(
            layer=Layer.character,
            priority=10,
            text="A calm spirit.",
            estimated_tokens=4,
            source="stub:char",
        )
        pipeline = ContextPipeline(providers=[_StubProvider("stub", [contribution])])

        result = await pipeline.assemble(
            _make_request(),
            budget_by_layer={Layer.character: 100},
        )

        assert result.budget.by_layer[Layer.character] == "A calm spirit."
        assert len(result.outcomes) == 1
        outcome = result.outcomes[0]
        assert outcome.provider_id == "stub"
        assert outcome.status == "ok"
        assert outcome.error_message is None
        assert outcome.contributions == [contribution]

    @pytest.mark.asyncio
    async def test_multiple_providers_run_concurrently(self) -> None:
        """Providers fan out under the TaskGroup.

        Two providers each sleeping 0.1s should finish in well under
        0.2s total if they truly run in parallel.
        """
        p1 = _StubProvider(
            "p1",
            [
                Contribution(
                    layer=Layer.character,
                    priority=10,
                    text="p1",
                    estimated_tokens=1,
                    source="p1",
                )
            ],
            sleep_s=0.1,
        )
        p2 = _StubProvider(
            "p2",
            [
                Contribution(
                    layer=Layer.content,
                    priority=10,
                    text="p2",
                    estimated_tokens=1,
                    source="p2",
                )
            ],
            sleep_s=0.1,
        )
        pipeline = ContextPipeline(providers=[p1, p2])

        start = asyncio.get_event_loop().time()
        result = await pipeline.assemble(
            _make_request(),
            budget_by_layer={Layer.character: 100, Layer.content: 100},
        )
        elapsed = asyncio.get_event_loop().time() - start

        # Sequential would be ~0.2s; concurrent should be well under that.
        assert elapsed < 0.18, f"expected concurrent execution, took {elapsed:.3f}s"

        # Both contributions made it through.
        assert result.budget.by_layer.get(Layer.character) == "p1"
        assert result.budget.by_layer.get(Layer.content) == "p2"
        assert {o.provider_id for o in result.outcomes} == {"p1", "p2"}
        assert all(o.status == "ok" for o in result.outcomes)


# ---------------------------------------------------------------------------
# Error handling — failing providers don't poison the pipeline
# ---------------------------------------------------------------------------


class TestFailingProviderIsIsolated:
    @pytest.mark.asyncio
    async def test_raising_provider_does_not_prevent_others(self) -> None:
        good = _StubProvider(
            "good",
            [
                Contribution(
                    layer=Layer.character,
                    priority=10,
                    text="Aria.",
                    estimated_tokens=2,
                    source="good",
                )
            ],
        )
        bad = _FailingProvider("bad", RuntimeError("kaboom"))
        pipeline = ContextPipeline(providers=[bad, good])

        result = await pipeline.assemble(
            _make_request(),
            budget_by_layer={Layer.character: 100},
        )

        assert result.budget.by_layer[Layer.character] == "Aria."

        by_id = {o.provider_id: o for o in result.outcomes}
        assert by_id["good"].status == "ok"
        assert by_id["bad"].status == "error"
        assert by_id["bad"].contributions == []
        assert by_id["bad"].error_message is not None
        assert "kaboom" in by_id["bad"].error_message

    @pytest.mark.asyncio
    async def test_slow_provider_past_deadline_is_dropped(self) -> None:
        good = _StubProvider(
            "good",
            [
                Contribution(
                    layer=Layer.character,
                    priority=10,
                    text="fast",
                    estimated_tokens=1,
                    source="good",
                )
            ],
        )
        # A provider that sleeps longer than its own deadline.
        slow = _StubProvider(
            "slow",
            [
                Contribution(
                    layer=Layer.content,
                    priority=10,
                    text="too late",
                    estimated_tokens=2,
                    source="slow",
                )
            ],
            deadline_s=0.05,
            sleep_s=0.5,
        )
        pipeline = ContextPipeline(providers=[good, slow])

        result = await pipeline.assemble(
            _make_request(),
            budget_by_layer={Layer.character: 100, Layer.content: 100},
        )

        by_id = {o.provider_id: o for o in result.outcomes}
        assert by_id["good"].status == "ok"
        assert by_id["slow"].status == "timeout"
        assert by_id["slow"].contributions == []

        # The slow provider's content did not make it into the result.
        assert result.budget.by_layer.get(Layer.character) == "fast"
        assert result.budget.by_layer.get(Layer.content) is None


# ---------------------------------------------------------------------------
# Pre-processors
# ---------------------------------------------------------------------------


class TestPreProcessors:
    @pytest.mark.asyncio
    async def test_pre_processor_mutates_request_seen_by_provider(self) -> None:
        pre = _StubPreProcessor("pre", "rewritten utterance")
        provider = _StubProvider("p", [])

        pipeline = ContextPipeline(
            providers=[provider],
            pre_processors=[pre],
        )
        result = await pipeline.assemble(_make_request(), budget_by_layer={})

        assert provider.last_request is not None
        assert provider.last_request.utterance == "rewritten utterance"
        # The final PipelineOutput.request reflects the pre-processed version.
        assert result.request.utterance == "rewritten utterance"

    @pytest.mark.asyncio
    async def test_multiple_pre_processors_run_in_registration_order(self) -> None:
        first = _StubPreProcessor("first", "pass-1")
        second = _StubPreProcessor("second", "pass-2")  # overrides first
        provider = _StubProvider("p", [])

        pipeline = ContextPipeline(
            providers=[provider],
            pre_processors=[first, second],
        )
        result = await pipeline.assemble(_make_request(), budget_by_layer={})
        # second ran after first, so it wins
        assert result.request.utterance == "pass-2"

    @pytest.mark.asyncio
    async def test_preprocessor_contributions_reach_budgeter(self) -> None:
        """Pre-processor contributions are merged into the budgeter input.

        A Contribution stashed on the request via the
        ``preprocessor_contributions`` field rides through alongside
        provider contributions on the way to the budgeter.
        """
        pre_contribution = Contribution(
            layer=Layer.depth_inject,
            priority=50,
            text="hidden chain of thought",
            estimated_tokens=4,
            source="stepped_thinking",
        )
        provider_contribution = Contribution(
            layer=Layer.character,
            priority=100,
            text="A calm spirit.",
            estimated_tokens=4,
            source="char",
        )
        pipeline = ContextPipeline(
            providers=[_StubProvider("p", [provider_contribution])],
            pre_processors=[_ContributingPreProcessor("pre", pre_contribution)],
        )
        result = await pipeline.assemble(
            _make_request(),
            budget_by_layer={Layer.character: 100, Layer.depth_inject: 100},
        )

        # Both layers populated.
        assert result.budget.by_layer[Layer.character] == "A calm spirit."
        assert result.budget.by_layer[Layer.depth_inject] == "hidden chain of thought"


# ---------------------------------------------------------------------------
# Post-processors
# ---------------------------------------------------------------------------


class TestPostProcessors:
    """``run_post_processors`` is the call-site for PostProcessors.

    The pipeline deliberately does *not* own the LLM call (the bot
    layer still does that, so TTS and history persistence can stay
    clustered together). Instead, the pipeline exposes a narrow
    ``run_post_processors`` method that the bot invokes on the LLM's
    reply before writing it to history or TTS.

    Post-processors run in **reverse** registration order so that the
    most-recently-registered processor (the one the operator added
    last / the outermost wrapper) is the last to see the reply. Each
    processor that raises or times out degrades to a no-op: its
    exception is caught and the previous stage's text is passed
    through, so one buggy processor never swallows a reply entirely.
    """

    @pytest.mark.asyncio
    async def test_no_post_processors_returns_reply_unchanged(self) -> None:
        pipeline = ContextPipeline(providers=[])
        result = await pipeline.run_post_processors("original", _make_request())
        assert result == "original"

    @pytest.mark.asyncio
    async def test_single_post_processor_sees_reply(self) -> None:
        post = _RecordingPostProcessor("p", "A")
        pipeline = ContextPipeline(providers=[], post_processors=[post])

        result = await pipeline.run_post_processors("hello", _make_request())

        assert post.seen == ["hello"]
        assert result == "hello|A"

    @pytest.mark.asyncio
    async def test_post_processors_run_in_reverse_registration_order(self) -> None:
        """Later processors run first so they wrap earlier ones symmetrically."""
        inner = _RecordingPostProcessor("inner", "inner")
        outer = _RecordingPostProcessor("outer", "outer")
        pipeline = ContextPipeline(providers=[], post_processors=[inner, outer])

        result = await pipeline.run_post_processors("hi", _make_request())

        # Outer runs first (sees the raw reply), then inner sees outer's output.
        assert outer.seen == ["hi"]
        assert inner.seen == ["hi|outer"]
        assert result == "hi|outer|inner"

    @pytest.mark.asyncio
    async def test_failing_post_processor_is_skipped(self) -> None:
        """A raising post-processor degrades to a no-op for just that stage."""
        post = _RecordingPostProcessor("good", "tag")
        bad = _FailingPostProcessor("bad", RuntimeError("kaboom"))
        pipeline = ContextPipeline(
            providers=[],
            post_processors=[post, bad],
        )

        result = await pipeline.run_post_processors("hi", _make_request())

        # bad runs first (reverse order), raises, gets skipped; then post runs
        # and sees the untransformed text.
        assert post.seen == ["hi"]
        assert result == "hi|tag"


# ---------------------------------------------------------------------------
# Pre-processor isolation — Protocol-declared PreProcessorError is caught
# ---------------------------------------------------------------------------


class TestPreProcessorIsolation:
    """Isolate pre-processors that raise ``PreProcessorError``; propagate the rest.

    Any other exception type is treated as a contract violation and
    propagates out of ``assemble`` — a future refactor that broadens
    the caught type to ``Exception`` will break the second test.
    """

    @pytest.mark.asyncio
    async def test_assemble_continues_when_preprocessor_raises_preprocessor_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A pre-processor that raises ``PreProcessorError`` is skipped.

        The following pre-processor still runs and sees the *unmodified*
        upstream request (i.e. it does not see any partial mutation
        from the raising processor), and ``assemble`` still returns a
        valid ``PipelineOutput``. A warning is logged with the
        processor id.
        """
        bad = _FailingPreProcessor("bad", PreProcessorError("nope"))
        downstream = _StubPreProcessor("downstream", "after-bad")
        provider = _StubProvider("p", [])

        pipeline = ContextPipeline(
            providers=[provider],
            pre_processors=[bad, downstream],
        )

        with caplog.at_level("WARNING", logger="familiar_connect.context.pipeline"):
            result = await pipeline.assemble(_make_request(), budget_by_layer={})

        # Pipeline ran to completion and the downstream pre-processor's
        # mutation is visible on the final request.
        assert isinstance(result, PipelineOutput)
        assert result.request.utterance == "after-bad"
        # The downstream processor saw the upstream request unchanged —
        # the raising processor did not corrupt the chain.
        assert provider.last_request is not None
        assert provider.last_request.utterance == "after-bad"

        # A warning was logged naming the raising processor.
        assert any(
            "bad" in record.getMessage() and record.levelname == "WARNING"
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_assemble_propagates_contract_violations(self) -> None:
        """A non-``PreProcessorError`` raise from a pre-processor crashes the pipeline.

        This pins the "contract violations surface loudly" guarantee:
        a future refactor that broadens the pipeline catch to
        ``Exception`` would break this test.
        """
        bad = _FailingPreProcessor("bad", RuntimeError("kaboom"))
        pipeline = ContextPipeline(
            providers=[_StubProvider("p", [])],
            pre_processors=[bad],
        )

        with pytest.raises(RuntimeError, match="kaboom"):
            await pipeline.assemble(_make_request(), budget_by_layer={})
