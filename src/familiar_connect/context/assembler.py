"""Prompt assembler.

Composes :class:`Layer` contributions into a system prompt + recent
history messages, with per-layer in-process caching keyed on
``invalidation_key``. See plan § Design.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect.context.layers import RagContextLayer, RecentHistoryLayer

if TYPE_CHECKING:
    from familiar_connect.context.layers import Layer
    from familiar_connect.llm import Message


@dataclass(frozen=True)
class AssemblyContext:
    """Inputs the assembler passes to every layer.

    :param viewer_mode: ``"voice"`` or ``"text"`` — selects
        :class:`OperatingModeLayer` output and may affect layer order.
    :param guild_id: Discord guild scoping per-guild nicknames; pass
        ``None`` for DMs or non-Discord platforms.
    """

    familiar_id: str
    channel_id: int | None
    viewer_mode: str = "text"
    guild_id: int | None = None


@dataclass
class AssembledPrompt:
    """Output of :meth:`Assembler.assemble`."""

    system_prompt: str = ""
    recent_history: list[Message] = field(default_factory=list)


class Assembler:
    """Layer composer with per-layer memoization.

    Layer order is preserved from construction. ``invalidation_key``
    per layer controls cache reuse — two assemble calls with the same
    context and unchanged layer keys return the same text without
    re-running :meth:`Layer.build`.
    """

    def __init__(self, *, layers: list[Layer]) -> None:
        self._layers: list[Layer] = list(layers)
        # key: (layer.name, invalidation_key) -> rendered text
        self._cache: dict[tuple[str, str], str] = {}

    def set_rag_cue(self, cue: str) -> None:
        """Forward *cue* to the first :class:`RagContextLayer`, if any.

        Lets the responder set the retrieval query without taking a
        direct handle on the layer object.
        """
        for layer in self._layers:
            if isinstance(layer, RagContextLayer):
                layer.set_current_cue(cue)
                return

    async def assemble(self, ctx: AssemblyContext) -> AssembledPrompt:
        sections: list[str] = []
        recent: list[Message] = []

        for layer in self._layers:
            if isinstance(layer, RecentHistoryLayer):
                recent = await layer.recent_messages(ctx)
                continue

            key = (layer.name, layer.invalidation_key(ctx))
            if key in self._cache:
                text = self._cache[key]
            else:
                text = await layer.build(ctx)
                self._cache[key] = text
            if text:
                sections.append(text)

        system_prompt = "\n\n".join(sections)
        return AssembledPrompt(
            system_prompt=system_prompt,
            recent_history=recent,
        )
