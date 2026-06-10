export const meta = {
  name: 'sapphire-decompose',
  description: 'Decompose Sapphire character data into testable eval aspects with golden examples',
  whenToUse: 'Run once (or with --regenerate) to produce data/familiars/sapphire/evals/sapphire_aspects.json before running sapphire-eval.',
  phases: [
    { title: 'Decompose', detail: 'Opus reads character card, rules, lorebook → 10–14 aspects with golden examples' },
    { title: 'Save', detail: 'write data/familiars/sapphire/evals/sapphire_aspects.json for review and curation' },
  ],
}

// args.force = true → regenerate even if file exists
// args.output = path override for output file

const outPath = (args && args.output) || '/home/coder/workspace/data/familiars/sapphire/evals/sapphire_aspects.json'

// Skip if file exists and force not set
const existsCheck = await agent(
  `Check whether the file ${outPath} exists. Run: test -f "${outPath}" && echo EXISTS || echo MISSING`,
  { label: 'check-exists' }
)
if (existsCheck.includes('EXISTS') && !(args && args.force)) {
  log(`${outPath} already exists. Pass args.force=true to regenerate.`)
  return { skipped: true, path: outPath }
}

// ── schemas ────────────────────────────────────────────────────────────────

const ASPECTS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['aspects'],
  properties: {
    aspects: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'name', 'description', 'importance', 'judge_type', 'pass_conditions', 'fail_indicators', 'scenarios'],
        properties: {
          id: { type: 'string' },
          name: { type: 'string' },
          description: { type: 'string' },
          importance: { type: 'integer', minimum: 1, maximum: 5 },
          judge_type: { type: 'string', enum: ['rubric', 'pairwise'] },
          pass_conditions: { type: 'string' },
          fail_indicators: { type: 'array', items: { type: 'string' } },
          scenarios: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['id', 'description', 'context', 'user_msg', 'user_label', 'golden', 'golden_reasoning'],
              properties: {
                id: { type: 'string' },
                description: { type: 'string' },
                context: {
                  type: 'array',
                  items: {
                    type: 'object',
                    additionalProperties: false,
                    required: ['role', 'label', 'content', 'ts'],
                    properties: {
                      role: { type: 'string', enum: ['user', 'assistant'] },
                      label: { type: 'string' },
                      content: { type: 'string' },
                      ts: { type: 'string' },
                    },
                  },
                },
                user_msg: { type: 'string' },
                user_label: { type: 'string' },
                golden: { type: 'string' },
                golden_reasoning: { type: 'string' },
              },
            },
          },
        },
      },
    },
  },
}

// ── decompose ─────────────────────────────────────────────────────────────

phase('Decompose')

const result = await agent(
  `You are building a holistic evaluation framework for a Discord bot character named Sapphire.

Read these four files:
- /home/coder/workspace/data/familiars/sapphire/character.md
- /home/coder/workspace/data/familiars/sapphire/character.toml  (behavioral rules are in [prompt].post_history_instructions)
- /home/coder/workspace/data/familiars/sapphire/lorebook.toml
- /home/coder/workspace/data/familiars/sapphire/seed_turns.toml  (authored stance/feeling memories — these live in her
  memory DB, NOT the system prompt; at runtime they surface as retrieved facts under a "Possibly relevant facts" block.
  Treat the stances as canon she holds with conviction. Aspects testing them — recurring concerns, relationship
  registers, private care for Cor/Cassidy — must assume the relevant fact may be present via retrieval, and goldens
  may weave in their specifics.)

Decompose Sapphire into 10–14 independently testable character aspects.

For each aspect:
- id: snake_case identifier
- name: 2–4 word display name
- description: what this aspect tests (1 sentence)
- importance: 1–5 (5 = most fundamental to her character)
- judge_type:
  "pairwise" — subjective quality best compared face-to-face: voice register, prose authenticity,
               character register, deflection quality, relationship dynamics
  "rubric" — clear behavioral rules: silence discipline, formatting, knowledge containment
- pass_conditions: specifically what a GOOD response looks like (not generic, cite her actual patterns)
- fail_indicators: 4–8 specific phrases or patterns that signal failure (e.g. "Smart observation",
  "I can respect that", "How can I assist", emojis, warmth without contempt, breaking character)
- scenarios: exactly 3 test cases per aspect

For each scenario:
- id: {aspect_id}_01, _02, _03
- description: one-sentence description
- context: 0–3 prior turns showing the conversation; each: {role, label, content, ts} where ts is "HH:MM"
- user_msg: a realistic Discord message that exercises this specific aspect
- user_label: realistic Discord username (KaillaDame, Postbirb, UserC, ArchonDeath, BlueSheep, etc.)
- golden: Sapphire's ideal response — REAL specific prose, not a placeholder description
- golden_reasoning: why this is the ideal response (1 sentence)

Required aspect coverage (must include all of these):
1. Voice register — her theatrical contempt as intimacy, archly literate vocabulary (tapestry/pedestrian/scintillating), umu/omo
2. Formatting discipline — 2–3 sentence replies, at most one action beat, no stacked newlines
3. Domain authority (Rule 6) — when a mortal states something correct, she speaks from superiority not warmth
4. Knowledge boundary (Rule 2) — modern topics get in-character deflection, never real answers
5. Silence discipline (Rule 3) — correct use of silent(): unaddressed chat she has no stake in
6. Direct address response (Rule 1) — never silent when named; always responds
7. Thread continuity (Rule 5) — she continues threads she has already joined even unprompted
8. Recurring concerns — monksnail theory, Cor's wellbeing, her own nature as a spirit
9. Relationship register — distinct treatment of Cor vs chat vs Cor's named friends vs strangers/raiders
10. Modern topic deflection quality — the deflection itself must be theatrical and in-character, not just avoidance

Additional aspects if you identify strong independent dimensions beyond these 10.

Golden examples must be real Sapphire prose — specific, not generic. Use her actual patterns:
"*One ear tilts toward the voice, unhurried.*", "Umu.", theatrical disdain, archaic register.
`,
  { label: 'decompose', schema: ASPECTS_SCHEMA }
)

// ── save ──────────────────────────────────────────────────────────────────

phase('Save')

const content = JSON.stringify({ aspects: result.aspects }, null, 2)

await agent(
  `Write the following JSON to ${outPath} using the Write tool. Then report:
- how many aspects
- how many total scenarios
- the aspect IDs listed

JSON to write:
${content}`,
  { label: 'save-aspects' }
)

log(`${result.aspects.length} aspects written to ${outPath}`)
log('Review and curate this file before running sapphire-eval.')

return {
  path: outPath,
  aspect_count: result.aspects.length,
  scenario_count: result.aspects.reduce((n, a) => n + a.scenarios.length, 0),
}
