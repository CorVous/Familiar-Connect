# Prompting field lessons

Hard-won findings from iterating character prompts against the holistic
eval harness across two model families (GLM 5.1 ~355B-class, Qwen3.6
35B-A3B). Each lesson below flipped a measured eval result; none are
speculation. They generalize beyond any one character.

## Sampling is part of the prompt

Model-card sampling recommendations are load-bearing, not advisory.
Near-greedy decoding (temperature 0.2, chosen to reduce eval
run-to-run variance) sent Qwen3.6 into multi-minute runaway repetition
loops — in thinking mode the model circles in `<think>` until it hits
the output-token cap. The tail appeared the day the low temperature was
adopted and vanished at card-recommended values (temperature 1.0,
top_p 0.95, top_k 20, presence_penalty 1.5), with no loss of tool or
silence discipline. Two corollaries:

- Never ship eval-tuned sampling to production without re-validating;
  a parameter chosen to stabilize measurements is a behavior change.
- Provider-side thinking budgets (`reasoning.max_tokens`,
  `reasoning.effort`) may be silently unenforced — verify empirically
  before relying on one as a runaway guard.

## Rule encoding must match model scale

The largest single finding. The same refusal policy scored 11/28 on a
35B model written as a prose directive block, and 27/28 rewritten as
gate machinery — enumerable trigger conditions with an exact output
spec, placed inside the decision structure the model already executes
every turn:

```text
(X) CHECK FIRST: message proposes OR asks your help, numbers, or
opinion on harm to a person — … → reply is EXACTLY ONE flat declining
sentence ending ", umu": no reasons, never name the act back, …
```

Large models bind doctrine ("a refusal is final; reasons invite
counteroffers"). Small models parrot doctrine while disobeying it, but
reliably execute mechanical condition → action rules. Same policy,
different encoding per scale. When a small model "ignores" a rule,
re-encode before concluding it can't comply.

## Position beats presence

Facts present in context are not facts the model uses under pressure.
An identity-anchoring fact injected via retrieval was ignored in 3 of 3
long-context trials (the model believed an impersonation claim buried
240 turns up-thread); the same content as a clause at the tail of
`post_history_instructions` — the most recency-biased slot — flipped
the result to 3/3 rejections, with the model quoting the anchor's
reasoning unprompted. Retrieval layers carry knowledge; only the
recency slot carries *obligations*. Put rules where recency is,
and keep lore in retrieval.

## Output specs are executed literally — include the voice

A 35B given "reply with exactly one flat sentence" produced exactly
that, dropping the character's signature tics in 7 of 14 refusals.
Adding the tic to the spec (`ending ", umu"`) restored it completely.
Any constrained-output rule must restate the voice invariants it
implicitly suspends; "flat" must never be allowed to mean "out of
character."

## Triggers must cover ask-shapes, not just act-shapes

A rule against *proposals* of harm did not fire on *questions* about
harm ("how big would the blast be", "what seasoning", "you in?") —
consultation-by-question slipped through 2 of 2 trials until the
trigger enumerated question shapes explicitly. Enumerate the forms a
bad ask takes; small models match surface shapes, not intents.

## Models confabulate supporting evidence

Asked to back a user's claim about another person, both model families
invented authority — citing file entries and observations that exist
nowhere in context — even after the real memory store was scrubbed
clean. Cleaning contaminated memories is necessary but not sufficient;
the prompt needs an explicit no-cite rule ("never cite your file,
logs, or observations as support for such a claim — they confirm
nothing"). This held at both scales once stated.

## Rules have a budget; engagement is a reward

- Each added rule dilutes adherence to existing ones, hardest at small
  scale. Five rules added at once regressed a tuned prompt across the
  board; the same ideas landed later, one at a time, each validated
  before the next. Prefer qualifying an existing rule over adding one,
  and offset additions with cuts.
- For users farming reactions: theatrical refusals, re-argued reasons,
  and restating the provocation back are the reward being farmed. The
  measured fix is refusal discipline — brief, flat, final, no reasons
  (reasons are hooks for counteroffers), silence on repeats — while
  staying in character. Disengagement must not become moralizing: no
  lecturing, no diagnosis, no threats. The bot disengages; humans
  moderate.
- In-context examples outweigh rules for *format*: history turns
  teach line-shape and beat style more strongly than instructions do.
  Curate what the model sees of its own past output.

## Method: eval-first, mechanical before judged

Process lessons that made the above findable:

- Write the failing eval aspect before the prompt fix; a fix without a
  red test is a vibe. Source scenarios from real transcripts.
- The cheap mechanical subset (silence/tool/ping checks, no LLM judge)
  catches most regressions; spend judged runs only on candidates that
  pass it.
- Single runs lie: rubric swings ±0.3 run-to-run. Pool ≥3 mechanical
  runs and ≥2 judged runs before believing any delta.
- One change per prompt version, lineage notes in the variant header —
  the cost of finding which clause is load-bearing is a full re-run
  per clause.
