"""Prompt composition — layered system-prompt assembly.

Each ``Layer`` owns one segment of the system prompt with its own
invalidation signal. The :class:`Assembler` composes non-empty layers
into :class:`familiar_connect.llm.SystemPromptLayers`. See plan
§ Design.4 *Prompt composition*.
"""

from __future__ import annotations

from familiar_connect.context.assembler import (
    AssembledPrompt,
    Assembler,
    AssemblyContext,
)
from familiar_connect.context.layers import (
    CharacterCardLayer,
    ConversationSummaryLayer,
    CoreInstructionsLayer,
    CrossChannelContextLayer,
    Layer,
    LorebookEntry,
    LorebookLayer,
    OperatingModeLayer,
    PeopleDossierLayer,
    RagContextLayer,
    RecentHistoryLayer,
    ReflectionLayer,
)

__all__ = [
    "AssembledPrompt",
    "Assembler",
    "AssemblyContext",
    "CharacterCardLayer",
    "ConversationSummaryLayer",
    "CoreInstructionsLayer",
    "CrossChannelContextLayer",
    "Layer",
    "LorebookEntry",
    "LorebookLayer",
    "OperatingModeLayer",
    "PeopleDossierLayer",
    "RagContextLayer",
    "RecentHistoryLayer",
    "ReflectionLayer",
]
