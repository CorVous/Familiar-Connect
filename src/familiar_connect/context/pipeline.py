"""The ContextPipeline — the asyncio orchestrator.

Walks the registered pre-processors, fans the providers out under a
scoped ``asyncio.TaskGroup`` with per-provider deadlines, collects
their Contributions, and hands everything to the Budgeter. Individual
provider failures and deadline misses are isolated so one misbehaving
provider never poisons the rest of the pipeline.

The pipeline deliberately does **not** call the main LLM itself. The
bot layer owns the LLM call so that TTS fan-out and history
persistence stay clustered together; the pipeline's
:meth:`ContextPipeline.run_post_processors` hook lets the bot fire
post-processors against the resulting reply without having to
re-implement the reverse-order, per-stage failure isolation every
call site would otherwise need.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect.context.budget import Budgeter, BudgetResult
from familiar_connect.context.protocols import PreProcessorError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from familiar_connect.context.protocols import (
        ContextProvider,
        PostProcessor,
        PreProcessor,
    )
    from familiar_connect.context.types import (
        ContextRequest,
        Contribution,
        Layer,
    )

_logger = logging.getLogger(__name__)

ProviderStatus = str  # "ok" | "error" | "timeout"


@dataclass
class ProviderOutcome:
    """Per-provider execution record for logging and the dashboard.

    One of these exists for every provider the pipeline ran, regardless
    of whether it succeeded, errored, or timed out. The ``status`` field
    is ``"ok"``, ``"error"``, or ``"timeout"``; ``error_message`` is
    populated for the non-ok cases.
    """

    provider_id: str
    duration_s: float
    contributions: list[Contribution]
    status: ProviderStatus
    error_message: str | None = None


@dataclass
class PipelineOutput:
    """Result of a single pipeline run.

    :param request: The ``ContextRequest`` as seen by the providers,
        i.e. after all pre-processors have run. This is the canonical
        "what did the pipeline actually see" record for logs.
    :param budget: The budgeter's output — per-layer text and the list
        of dropped / truncated contributions.
    :param outcomes: One entry per provider that was executed.
    """

    request: ContextRequest
    budget: BudgetResult
    outcomes: list[ProviderOutcome] = field(default_factory=list)


class ContextPipeline:
    """Assemble context for a single reply turn.

    The pipeline is stateless across turns — construct it once with the
    active providers / pre-processors / post-processors, then call
    :meth:`assemble` per request. Providers are run concurrently inside
    a scoped ``asyncio.TaskGroup`` with per-provider deadlines; a
    provider that misses its deadline or raises is recorded in the
    output and does not affect other providers.
    """

    def __init__(
        self,
        providers: Sequence[ContextProvider],
        pre_processors: Sequence[PreProcessor] = (),
        post_processors: Sequence[PostProcessor] = (),
        budgeter: Budgeter | None = None,
    ) -> None:
        self._providers = list(providers)
        self._pre_processors = list(pre_processors)
        self._post_processors = list(post_processors)
        self._budgeter = budgeter or Budgeter()

    async def assemble(
        self,
        request: ContextRequest,
        budget_by_layer: dict[Layer, int],
    ) -> PipelineOutput:
        """Run the pipeline for *request* and return a :class:`PipelineOutput`.

        :param request: The request built from the incoming event.
        :param budget_by_layer: Per-layer token budgets to hand to the
            budgeter.
        """
        # 1. Pre-processors run sequentially; each sees the previous one's
        # output. A pre-processor that raises ``PreProcessorError`` is
        # skipped — ``processed`` is not updated, so the next stage sees
        # the last successful value. Any other exception escapes this
        # loop and crashes the pipeline: that is a Protocol-contract
        # violation and is intentionally surfaced loudly rather than
        # silently masked by a blanket ``except``.
        processed = request
        for pre in self._pre_processors:
            try:
                processed = await pre.process(processed)
            except PreProcessorError as exc:
                _logger.warning(
                    "pre-processor %s raised %s: %s; skipping",
                    pre.id,
                    type(exc).__name__,
                    exc,
                )
                continue

        # 2. Providers fan out concurrently under a single TaskGroup.
        outcomes = await self._run_providers(processed)

        # 3. Everything that came back goes to the budgeter — both
        # provider outputs and any contributions pre-processors stashed
        # on the request via dataclasses.replace().
        all_contributions: list[Contribution] = list(
            processed.preprocessor_contributions
        )
        for outcome in outcomes:
            all_contributions.extend(outcome.contributions)

        budget_result = self._budgeter.fill(all_contributions, budget_by_layer)

        return PipelineOutput(
            request=processed,
            budget=budget_result,
            outcomes=outcomes,
        )

    async def run_post_processors(
        self,
        reply_text: str,
        request: ContextRequest,
    ) -> str:
        """Run every registered post-processor against *reply_text*.

        Processors run in **reverse** registration order so the
        most-recently-registered processor is the outermost wrapper
        and sees every earlier transformation. A processor that
        raises is caught, logged, and skipped — its input is passed
        through to the next stage unchanged so one buggy processor
        never eats a reply entirely.
        """
        current = reply_text
        for post in reversed(self._post_processors):
            try:
                current = await post.process(current, request)
            except Exception as exc:  # noqa: BLE001 — isolation by design
                _logger.warning(
                    "post-processor %s raised %s: %s; passing input through",
                    post.id,
                    type(exc).__name__,
                    exc,
                )
        return current

    async def _run_providers(
        self,
        request: ContextRequest,
    ) -> list[ProviderOutcome]:
        """Run every provider concurrently with its own deadline."""
        if not self._providers:
            return []

        # Each provider's outcome is written into this list at its fixed
        # index so the caller sees a deterministic order.
        outcomes: list[ProviderOutcome | None] = [None] * len(self._providers)

        async with asyncio.TaskGroup() as tg:
            for idx, provider in enumerate(self._providers):
                tg.create_task(
                    self._run_single_provider(idx, provider, request, outcomes)
                )

        # At this point every slot is filled (_run_single_provider never
        # leaves a None behind — it catches everything).
        return [o for o in outcomes if o is not None]

    async def _run_single_provider(
        self,
        idx: int,
        provider: ContextProvider,
        request: ContextRequest,
        outcomes: list[ProviderOutcome | None],
    ) -> None:
        """Invoke one provider and record its outcome.

        This method **must not raise** — otherwise it would propagate
        out of the TaskGroup and cancel sibling providers. All errors
        are captured as ``ProviderOutcome``s instead.
        """
        loop = asyncio.get_event_loop()
        start = loop.time()
        try:
            async with asyncio.timeout(provider.deadline_s):
                contributions = await provider.contribute(request)
        except TimeoutError:
            duration = loop.time() - start
            _logger.warning(
                "context provider %s timed out after %.3fs (deadline %.3fs)",
                provider.id,
                duration,
                provider.deadline_s,
            )
            outcomes[idx] = ProviderOutcome(
                provider_id=provider.id,
                duration_s=duration,
                contributions=[],
                status="timeout",
                error_message=(f"exceeded deadline of {provider.deadline_s:.3f}s"),
            )
            return
        except Exception as exc:  # noqa: BLE001 — the whole point is to isolate.
            duration = loop.time() - start
            _logger.warning(
                "context provider %s raised %s: %s",
                provider.id,
                type(exc).__name__,
                exc,
            )
            outcomes[idx] = ProviderOutcome(
                provider_id=provider.id,
                duration_s=duration,
                contributions=[],
                status="error",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            return

        duration = loop.time() - start
        outcomes[idx] = ProviderOutcome(
            provider_id=provider.id,
            duration_s=duration,
            contributions=list(contributions),
            status="ok",
            error_message=None,
        )
