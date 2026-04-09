"""Concrete pre/post-processor implementations.

Each module in this package implements one processor that conforms
to either the :class:`PreProcessor` or :class:`PostProcessor`
Protocol from ``familiar_connect.context.protocols``. Pre-processors
mutate the outgoing :class:`ContextRequest` (typically by stashing
contributions on its ``preprocessor_contributions`` field);
post-processors mutate the main LLM's reply text before it reaches
TTS.
"""
