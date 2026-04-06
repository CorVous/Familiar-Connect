# Chattiness Personality Description

## Overview

The chattiness preset controls the mechanical frequency of responses, but not the *character* behind them. A personality-based chattiness description would let the familiar's reason for being quiet or talkative come through in how it actually speaks.

## Concept

A familiar set to **Moderate** could feel very different depending on its personality:

- A shy familiar might hesitate, hedge, or apologize for speaking up unprompted
- An arrogant familiar might act like it's doing the room a favour by chiming in
- A curious one might jump in with a question rather than a statement

The preset governs *when* it responds. The personality description governs *how it feels* about responding.

## Implementation Idea

Add an optional free-text field to the familiar's character card — something like "chattiness personality" or just let it emerge naturally from the main personality description. The character card already has a personality summary field (borrowing from TavernAI Character Card V2); the familiar's relationship to conversation could simply be part of that.

No separate field may be needed — if the personality description says "you are reserved and only speak when you have something meaningful to contribute," the LLM will naturally reflect that in the tone and framing of unprompted responses.

## Relationship to the LLM-Based Chattiness Future Feature

If a smaller model is eventually used to make response-worthiness decisions (see `plan.md`), the personality description becomes even more useful — it could be passed to that model as context so the decision of *whether* to respond is also personality-informed, not just mechanically threshold-based.

## Notes

- This is a soft feature — it requires no new system components, just thoughtful character card writing
- The preset and the personality description are complementary, not redundant
