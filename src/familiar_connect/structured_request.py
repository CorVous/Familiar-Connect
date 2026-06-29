"""Provisioning interface for structured LLM output — the request side.

:mod:`familiar_connect.structured_output` owns the PARSE side (turn a raw
reply str into JSON, tolerate garbage, never raise). This module owns the
REQUEST side: it is the single place a feature module says "I want this
shape back," and the only place that knows HOW we coerce a model into
giving it — the reply-shape contract wording, the retry-with-correction
loop, and the engineerable constants (max retries, token format).

Before this module each call site hand-typed its own ``Reply JSON only:
{...}`` string inside the feature code (see #167 / the #136 smell) and
none of them retried — a fumbled reply silently zeroed the work. Here a
caller declares a :class:`Schema` instead; :func:`render_contract` turns
it into the contract text appended to the prompt, and
:func:`request_structured` runs the call, parses to the declared shape,
and RE-ASKS with a clear correction when the model returns the wrong
thing. Domain validation (grounding rails, id filtering, dedup) stays in
the caller — this layer only guarantees "you got JSON of the declared
root shape, or you got nothing."

Strategy seam: today the format is JSON-in-prose and resilience is a
bounded re-ask. The contract renderer, the parser coupling, and
:data:`DEFAULT_MAX_RETRIES` are all isolated here, so swapping to TOML, to
tool-call arguments, or to a different retry budget changes this module
and nothing else — the point of #167.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from familiar_connect import log_style as ls
from familiar_connect.llm import Message
from familiar_connect.structured_output import coerce_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)

# THE retry knob (issue #167's "max retries"). One corrective re-ask after
# the first failure — enough to recover a model that fenced its JSON or
# added a sentence of preamble, without doubling cost on every healthy
# call. A call site with a different cost/latency budget overrides per
# call; this is the default to engineer against.
DEFAULT_MAX_RETRIES = 1


@dataclass(frozen=True)
class Field:
    """One field of a structured reply, rendered into the shape contract.

    ``placeholder`` is the token shown verbatim in the JSON skeleton the
    model copies — pick it to read as the value's type, e.g.
    ``'"<stance>"'`` for a string, ``"[<id>...]"`` for an int list,
    ``"<1-10>"`` for a bounded int. ``desc`` is an optional plain-language
    gloss rendered as a bullet under the skeleton for fields whose meaning
    isn't obvious from the key alone. ``required=False`` only changes the
    rendered wording (marks it optional) — enforcement is the caller's
    rails, never this layer.
    """

    name: str
    placeholder: str
    desc: str = ""
    required: bool = True


@dataclass(frozen=True)
class Schema:
    """Declarative shape of a structured LLM reply + how to parse it.

    Covers every shape the feature modules need today:

      * top-level ARRAY of items — ``root="array"`` (``container`` unused).
      * OBJECT wrapping one named list of items — ``root="object"`` with
        ``container="candidates"`` → ``{"candidates": [ {item}, ... ]}``.
      * flat OBJECT of fields — ``root="object"`` with ``container=None``
        → ``{ field: value, ... }``.

    In every case ``fields`` describes the *item* (the array element, the
    container element, or the flat object itself). ``empty_note`` and
    ``constraints`` append trailing contract lines ("Empty list when
    nothing stands out.", "Only use ids from the list below.").

    ``container`` is meaningful only for ``root="object"``; pairing it
    with ``root="array"`` is a programming error and raises.
    """

    fields: tuple[Field, ...]
    root: Literal["object", "array"] = "array"
    container: str | None = None
    empty_note: str = ""
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a container on an array root — a caller bug, not bad data."""
        if self.container is not None and self.root != "object":
            msg = "Schema.container is only valid when root='object'"
            raise ValueError(msg)


@dataclass(frozen=True)
class StructuredReply:
    """Outcome of a structured request.

    ``ok`` is the success signal; on failure (every attempt fumbled the
    shape) ``value`` is ``None`` and the caller degrades to its empty
    container, exactly as the pre-existing per-site code did. ``attempts``
    is how many LLM calls were spent (1 + the retries actually used) —
    useful for diagnostics and tuning :data:`DEFAULT_MAX_RETRIES`.
    """

    value: Any = None
    ok: bool = False
    attempts: int = 0


def render_contract(schema: Schema) -> str:
    """Render *schema* into the reply-shape contract appended to a prompt.

    Produces a literal JSON skeleton (what models lock onto) plus optional
    per-field bullets and any trailing constraints / empty-note. This is
    the single authoritative wording — no feature module hand-types a
    ``Reply JSON only: {...}`` string anymore.
    """
    item = _item_skeleton(schema.fields)
    if schema.root == "array":
        skeleton = f"[{item}, ...]"
    elif schema.container is not None:
        skeleton = f'{{"{schema.container}": [{item}, ...]}}'
    else:
        skeleton = item

    lines = [f"Reply with JSON only, no prose or code fences: {skeleton}"]
    bullets = [_field_bullet(f) for f in schema.fields if f.desc]
    if bullets:
        per_item = schema.root == "array" or schema.container is not None
        lines.append("Each item's fields:" if per_item else "Fields:")
        lines.extend(bullets)
    lines.extend(schema.constraints)
    if schema.empty_note:
        lines.append(schema.empty_note)
    return "\n".join(lines)


def _item_skeleton(fields: Sequence[Field]) -> str:
    """Render one ``{"name": <placeholder>, ...}`` JSON object skeleton."""
    inner = ", ".join(f'"{f.name}": {f.placeholder}' for f in fields)
    return "{" + inner + "}"


def _field_bullet(f: Field) -> str:
    """Render one field's explanatory bullet line."""
    opt = "" if f.required else " (optional)"
    return f"- `{f.name}`{opt}: {f.desc}"


async def request_structured(
    llm: LLMClient,
    *,
    messages: list[Message],
    schema: Schema,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> StructuredReply:
    """Ask *llm* for output matching *schema*; re-ask on a wrong shape.

    *messages* is the fully-built prompt — the caller appends
    :func:`render_contract`'s text to its system message so the contract
    stays co-located with the prompt (and inside any cache breakpoint).
    Each attempt is parsed via
    :func:`familiar_connect.structured_output.coerce_json` to the
    schema's root; a reply that is unparseable or the wrong root type is a
    shape failure, answered with a corrective follow-up that restates the
    contract, up to *max_retries* times.

    Returns the parsed root value (a ``dict`` for ``root="object"``, a
    ``list`` for ``root="array"``) with ``ok=True``, or
    ``StructuredReply(value=None, ok=False)`` when every attempt fumbled —
    the caller then degrades to its empty container. Domain validation
    (container key, ids, dedup) is the caller's job; this never inspects
    field values.

    Transport failures from :meth:`LLMClient.chat` (network, rate limit,
    HTTP errors) propagate unchanged — only SHAPE problems are retried or
    degraded, matching how the per-site code let worker loops own
    transport errors.
    """
    retries = max(0, max_retries)
    convo = list(messages)
    attempts = 0
    last_problem = ""

    for attempt in range(retries + 1):
        reply = await llm.chat(convo)
        attempts += 1
        content = reply.content_str
        problem = _shape_problem(content, schema)
        if problem is None:
            value = coerce_json(content, expect=schema.root).value
            return StructuredReply(value=value, ok=True, attempts=attempts)
        last_problem = problem
        if attempt < retries:
            convo = [
                *convo,
                Message(role="assistant", content=content),
                Message(role="user", content=_correction(problem, schema)),
            ]

    _log_degraded(llm, schema, attempts, last_problem)
    return StructuredReply(value=None, ok=False, attempts=attempts)


def _shape_problem(reply: str, schema: Schema) -> str | None:
    """Return why *reply* doesn't satisfy *schema*'s root, or ``None`` if it does.

    The string doubles as the model-facing correction reason, so it names
    the expected shape plainly.
    """
    result = coerce_json(reply, expect=schema.root)
    if not result.parsed_ok:
        return "the reply was not valid JSON"
    if schema.root == "object" and not isinstance(result.value, dict):
        return "the top-level value must be a JSON object ({...})"
    if schema.root == "array" and not isinstance(result.value, list):
        return "the top-level value must be a JSON array ([...])"
    return None


def _correction(problem: str, schema: Schema) -> str:
    """Build the corrective follow-up that re-states the contract."""
    return (
        f"Your previous reply could not be used: {problem}. Reply again "
        "with ONLY the JSON described below — no prose, no code fences, no "
        f"explanation.\n{render_contract(schema)}"
    )


def _log_degraded(llm: LLMClient, schema: Schema, attempts: int, problem: str) -> None:
    """Warn once when a request degrades to empty after exhausting retries."""
    slot = getattr(llm, "slot", None) or "-"
    _logger.warning(
        f"{ls.tag('Structured', ls.R)} "
        f"{ls.kv('degraded', schema.root, vc=ls.R)} "
        f"{ls.kv('slot', slot, vc=ls.LC)} "
        f"{ls.kv('attempts', str(attempts), vc=ls.LW)} "
        f"{ls.kv('problem', problem, vc=ls.LW)}"
    )
