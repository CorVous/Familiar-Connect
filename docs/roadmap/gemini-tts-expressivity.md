# Gemini TTS Expressivity (follow-on)

The Gemini TTS integration shipped with static, per-character performance
direction (Audio Profile / Scene / Director's Notes composed from six
`[tts]` fields). Two follow-on features would make the voice feel more alive:

## Dynamic audio-tag injection

**Motivation.** Gemini 3.1 Flash TTS supports 200+ inline audio tags
(`[laughs]`, `[whispers]`, `[short pause]`, `[gasp]`, `[panic]`, etc.)
embedded directly in the synthesis text. Today they only appear if the LLM
happens to include them in its prose reply. Systematic injection would make
emotional texture consistent and deliberate.

**Sketch.** Two viable approaches:

1. **Main-prose prompt tuning** — add a section to the familiar's character
   card (or the `post_process_style` recast prompt) instructing the LLM to
   annotate replies with Gemini audio tags when `[tts].provider = "gemini"`.
   Zero code; the LLM does the work. Risk: LLM may under- or over-annotate,
   and tags bleed into text-channel delivery.

2. **Lightweight post-processor** — a new `GeminiTagPostProcessor` that
   runs after `RecastPostProcessor` only on voice channels with a Gemini
   TTS client. Uses a small, cheap LLM call (e.g. `post_process_style`
   slot) to annotate the final reply text before synthesis. The processor
   strips tags before text-channel delivery.

**Open questions.**
- Which approach (prompt tuning vs. separate processor)?
- How to strip tags cleanly before text-channel output without a regex
  fragility surface?
- Should tags be applied to the full reply or only to specific emotional
  inflection points?

**Non-goals.** Generating custom (non-prebuilt) audio tags; that requires
Gemini API features not yet stable.

---

## Situational style prompts

**Motivation.** The current style prompt is composed once at startup from
static `character.toml` fields. A familiar's voice delivery doesn't change
when they're nervous, excited, or whispering in a tense scene — but it could.

**Sketch.** Thread the bot's current `MoodState` (already tracked in
`familiar_connect.mood`) into `GeminiTTSClient.synthesize` at call time.
A mapping from mood to style-prompt overrides (e.g. `mood=stressed` →
`pace = "clipped, shallow breaths"`) is defined in `character.toml` under
a new `[tts.gemini.mood_overrides.<mood>]` table. The client merges the
base style prompt with the active mood overrides before each synthesis call.

**Open questions.**
- Should overrides replace or extend the static fields?
- Does the composer stay in the factory, or move into the client so it can
  accept a mood argument?
- How do mood overrides interact with the interrupt-resume path (which
  synthesises remaining text mid-utterance)?

**Non-goals.** Per-*utterance* style prompts driven by LLM-classified
emotional content; that's a larger feature requiring a new call site.
