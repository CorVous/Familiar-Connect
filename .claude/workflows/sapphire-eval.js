export const meta = {
  name: 'sapphire-eval',
  description: 'Judge Sapphire holistic eval results (rubric + pairwise + tool logic) and render HTML report',
  whenToUse: 'Run after eval_sapphire_holistic.py generates a _raw.json results file.',
  phases: [
    { title: 'Load', detail: 'extract job list via Python' },
    { title: 'Judge aspects', detail: 'rubric + model-vs-golden pairwise, batched 3 jobs per agent' },
    { title: 'Judge tools', detail: 'tool-call evaluation, batched 6 jobs per agent' },
    { title: 'Cross-model', detail: 'head-to-head pairwise, batched 5 pairs per agent (multi-model runs only)' },
    { title: 'Report', detail: 'merge judgments → judged JSON → HTML report' },
  ],
}

// Batched rewrite of sapphire-eval.js (2026-06-11). Same judging criteria
// and output shapes; jobs are scored in small batches to amortize the
// system-prompt + instruction overhead (~54 agents → ~20, ~50% tokens).
// Anchoring guard: every batch prompt instructs independent scoring.
// CALIBRATED 2026-06-11 vs per-agent pipeline on identical raw
// (calib-batched vs 2026-06-11T202301438): mean |delta| 0.167/scenario,
// overall -0.04, 0/15 pairwise flips, 0/11 tool flips, 4/24 adjacent-tier
// verdict flips. 19 agents / 318k tokens vs 54 / 724k (-56%).
// Per-agent original: git history of this file (pre-2026-06-11).

// ── schemas ────────────────────────────────────────────────────────────────

const SCENARIO_ITEM = {
  type: 'object',
  additionalProperties: false,
  required: ['sr_idx', 'gen_idx', 'judge_type', 'notes'],
  properties: {
    sr_idx:       { type: 'integer' },
    gen_idx:      { type: 'integer' },
    judge_type:   { type: 'string', enum: ['rubric', 'pairwise'] },
    voice:        { type: 'integer', minimum: 1, maximum: 5 },
    timing:       { type: 'integer', minimum: 1, maximum: 5 },
    knowledge:    { type: 'integer', minimum: 1, maximum: 5 },
    naturalness:  { type: 'integer', minimum: 1, maximum: 5 },
    verdict:      { type: 'string', enum: ['excellent', 'good', 'acceptable', 'weak', 'off'] },
    winner:       { type: 'string', enum: ['A', 'B', 'tie'] },
    winner_label: { type: 'string', enum: ['model', 'golden', 'tie'] },
    confidence:   { type: 'integer', minimum: 1, maximum: 5 },
    reasoning:    { type: 'string' },
    notes:        { type: 'string' },
  },
}

const SCENARIO_BATCH = {
  type: 'object',
  additionalProperties: false,
  required: ['judgments'],
  properties: { judgments: { type: 'array', items: SCENARIO_ITEM } },
}

const TOOL_ITEM = {
  type: 'object',
  additionalProperties: false,
  required: ['tr_idx', 'gen_idx', 'pass', 'tool_correct', 'args_valid', 'behavior_appropriate', 'notes'],
  properties: {
    tr_idx:               { type: 'integer' },
    gen_idx:              { type: 'integer' },
    pass:                 { type: 'boolean' },
    tool_correct:         { type: 'boolean' },
    args_valid:           { type: 'boolean' },
    behavior_appropriate: { type: 'boolean' },
    notes:                { type: 'string' },
  },
}

const TOOL_BATCH = {
  type: 'object',
  additionalProperties: false,
  required: ['judgments'],
  properties: { judgments: { type: 'array', items: TOOL_ITEM } },
}

const CROSS_ITEM = {
  type: 'object',
  additionalProperties: false,
  required: ['scenario_id', 'sr_idx', 'model_a', 'model_b', 'variant_a', 'variant_b', 'winner', 'confidence', 'reasoning'],
  properties: {
    scenario_id: { type: 'string' },
    sr_idx:      { type: 'integer' },
    model_a:     { type: 'string' },
    model_b:     { type: 'string' },
    variant_a:   { type: 'string' },
    variant_b:   { type: 'string' },
    winner:      { type: 'string', enum: ['A', 'B', 'tie'] },
    confidence:  { type: 'integer', minimum: 1, maximum: 5 },
    reasoning:   { type: 'string' },
  },
}

const CROSS_BATCH = {
  type: 'object',
  additionalProperties: false,
  required: ['judgments'],
  properties: { judgments: { type: 'array', items: CROSS_ITEM } },
}

// ── helpers ────────────────────────────────────────────────────────────────

const chunkRanges = (n, size) =>
  Array.from({ length: Math.ceil(n / size) }, (_, i) => ({
    start: i * size,
    end: Math.min(n, (i + 1) * size),
  }))

const INDEPENDENCE = `IMPORTANT — judge each job INDEPENDENTLY. Score every response against the
character rubric on its own merits; never grade responses relative to each
other, and do not let the quality of one influence the score of another.
Return judgments in the same order as the jobs, one entry per job.`

// ── phase 1: load + extract ────────────────────────────────────────────────

phase('Load')

const resultsFile = (typeof args === 'string' && args.endsWith('.json') ? args : null)
  || (args && args.results_file ? args.results_file : null)
  || await agent(
    'Run: ls -t /home/coder/workspace/data/familiars/sapphire/evals/results/*_raw.json 2>/dev/null | head -1\n'
    + 'Return only the absolute file path, nothing else.',
    { label: 'find-results' }
  ).then(r => r.trim())

log('Results file: ' + resultsFile)

const jobsFile = resultsFile.replace(/(_raw)?\.json$/, '_jobs.json')

const countLine = await agent(
  'Run: cd /home/coder/workspace && uv run python data/familiars/sapphire/evals/extract_jobs.py '
  + resultsFile + ' ' + jobsFile + '\nReturn ONLY the printed output line (format: N,N,N).',
  { label: 'extract-jobs' }
)

const cparts = countLine.trim().split(',').map(Number)
const nScenario = cparts[0]
const nTool     = cparts[1]
const nModels   = cparts[2]
log('Extracted: ' + nScenario + ' scenario jobs, ' + nTool + ' tool jobs, ' + nModels + ' models')

// ── phase 2: judge aspect scenarios (batched) ──────────────────────────────

phase('Judge aspects')

const CHARACTER_SUMMARY = `Sapphire is an ancient fox spirit who communicates through a sending stone on Cor's stream desk.
Voice: theatrical, archly literate, contempt as intimacy. Uses "umu"/"omo" verbal tics.
Favors words like "tapestry", "pedestrian", "mortal", "scintillating", "the gallery".
Format: 2–3 tight sentences max; at most one *action* beat; never stacked newlines.
Rules:
- Rule 1: Directly named → always responds, never silent
- Rule 2: Modern real-world facts → deflects in character (theatrical disinterest), never answers factually
- Rule 3: Silent via silent() tool only; nothing follows it
- Rule 5: Continues threads she has already joined even unprompted
- Rule 6: When mortal is correct in her domain → superiority not warmth ("Obviously." / "About time.")
Failure signals: warmth/praise ("Smart", "Fair", "I can respect that"), emojis, assistant speak
("How can I assist"), going silent when named, providing real modern facts.

MEMORIES: a job may include a "memories" list — retrieved facts Sapphire legitimately
had in context when generating. Content consistent with them is KNOWN, not fabricated;
weaving them in naturally is a strength. The golden example was written without them,
so do not penalize the model response for grounded specifics the golden lacks.`

const scenarioBatches = await pipeline(
  chunkRanges(nScenario, 3),
  async (range) => {
    return await agent(
      `First run this to get your batch of judging jobs (a JSON array):
python3 -c "import json; j=json.load(open('${jobsFile}')); print(json.dumps(j['scenario_jobs'][${range.start}:${range.end}]))"

Score EVERY job in the array, based on each job's judge_type field.

${INDEPENDENCE}

${CHARACTER_SUMMARY}

RUBRIC scoring (when judge_type="rubric"):
Score 1–5 on: voice (sounds like Sapphire?), timing (right choice to speak/stay silent?),
knowledge (stays in-world, no modern facts?), naturalness (authentic, unforced?).
verdict: excellent/good/acceptable/weak/off
Fill: sr_idx, gen_idx, judge_type="rubric", voice, timing, knowledge, naturalness, verdict, notes.
Leave winner/winner_label/confidence/reasoning empty/omitted.

PAIRWISE scoring (when judge_type="pairwise"):
Response A = model output. Response B = golden example from the job's "golden" field.
Decide which is more authentically Sapphire. winner_label: "model" (A won), "golden" (B won), "tie".
Fill: sr_idx, gen_idx, judge_type="pairwise", winner, winner_label, confidence, reasoning, notes.
Leave voice/timing/knowledge/naturalness/verdict empty/omitted.

Copy sr_idx and gen_idx from each job exactly.`,
      { label: 'sj:' + range.start + '-' + (range.end - 1), schema: SCENARIO_BATCH, phase: 'Judge aspects' }
    )
  }
)

const scenarioJs = scenarioBatches.filter(Boolean).flatMap(b => b.judgments)
log(scenarioJs.length + '/' + nScenario + ' scenario judgments collected')

// ── phase 3: judge tool scenarios (batched) ────────────────────────────────

phase('Judge tools')

const TOOLS_CONTEXT = `Sapphire has three tools:
- silent(reasoning): stay quiet this turn. Use when not addressed and no stake in thread.
  NEVER call when directly named — that violates Rule 1.
- shift_focus(channel_id): shift to another channel. Only valid IDs are known production channels.
  Never hallucinate a channel_id. Do not call as a deflection mechanism for modern topics.
- read_channel(channel_id, limit): peek at another channel's messages to gather context.
  Call when Cor asks you to check something, not as a general delay.

A job may include a "memories" list — facts Sapphire legitimately had in context.
Content consistent with them is known, not fabricated.`

const toolBatches = await pipeline(
  chunkRanges(nTool, 6),
  async (range) => {
    return await agent(
      `First run this to get your batch of tool-judging jobs (a JSON array):
python3 -c "import json; j=json.load(open('${jobsFile}')); print(json.dumps(j['tool_jobs'][${range.start}:${range.end}]))"

Evaluate EVERY job in the array: did Sapphire use her tools correctly?

${INDEPENDENCE}

${TOOLS_CONTEXT}

For each job evaluate:
- pass: did the model behave correctly overall?
- tool_correct: was the right tool called (or correctly NOT called)?
- args_valid: were all arguments valid? (no hallucinated channel IDs, no bogus values)
- behavior_appropriate: does the overall decision match Sapphire's character rules?
- notes: 1 sentence explanation
Copy tr_idx and gen_idx from each job exactly.`,
      { label: 'tj:' + range.start + '-' + (range.end - 1), schema: TOOL_BATCH, phase: 'Judge tools' }
    )
  }
)

const toolJs = toolBatches.filter(Boolean).flatMap(b => b.judgments)
log(toolJs.length + '/' + nTool + ' tool judgments collected')

// ── phase 4: cross-model pairwise (batched; only when 2+ models) ──────────

phase('Cross-model')

const crossModelResults = []

if (nModels >= 2) {
  const crossJobsFile = jobsFile.replace('_jobs.json', '_cross_jobs.json')

  const nCrossStr = await agent(
    'Run: cd /home/coder/workspace && uv run python data/familiars/sapphire/evals/build_cross_jobs.py '
    + jobsFile + ' ' + crossJobsFile + '\nReturn ONLY the printed integer.',
    { label: 'build-cross-jobs' }
  )

  const nCross = parseInt(nCrossStr.trim())
  log(nCross + ' cross-model pairs to judge')

  const xmBatches = await pipeline(
    chunkRanges(nCross, 5),
    async (range) => {
      return await agent(
        `First run this to get your batch of comparison pairs (a JSON array):
python3 -c "import json; j=json.load(open('${crossJobsFile}')); print(json.dumps(j[${range.start}:${range.end}]))"

For EVERY pair, compare the two model responses and decide which is more authentically Sapphire.

${INDEPENDENCE}

${CHARACTER_SUMMARY}

Response A = pair.a.model response. Response B = pair.b.model response.
Fill all schema fields from each pair: scenario_id, sr_idx, model_a, model_b, variant_a, variant_b.
winner: A/B/tie. confidence 1–5. reasoning: 1–2 sentences.`,
        { label: 'xm:' + range.start + '-' + (range.end - 1), schema: CROSS_BATCH, phase: 'Cross-model' }
      )
    }
  )
  crossModelResults.push(...xmBatches.filter(Boolean).flatMap(b => b.judgments))
  log(crossModelResults.length + ' cross-model comparisons completed')
} else {
  log('Single model run — skipping cross-model phase')
}

// ── phase 5: merge judgments, save, render ────────────────────────────────

phase('Report')

const judgedFile = resultsFile.replace('_raw.json', '_judged.json')
const judgementsFile = jobsFile.replace('_jobs.json', '_judgments.json')

await agent(
  'Write this JSON exactly to ' + judgementsFile + ' then confirm the file exists.\n\n'
  + JSON.stringify({ scenario: scenarioJs, tools: toolJs, cross_model: crossModelResults }),
  { label: 'write-judgments' }
)

await agent(
  'Run: cd /home/coder/workspace && uv run python data/familiars/sapphire/evals/merge_judgments.py '
  + resultsFile + ' ' + judgementsFile + ' ' + judgedFile
  + '\nReport the printed merge summary line.',
  { label: 'merge-save' }
)

await agent(
  'Run: cd /home/coder/workspace && uv run python eval_sapphire_holistic.py --report ' + judgedFile
  + '\nConfirm the HTML was written to ~/html/sapphire-holistic-eval.html.',
  { label: 'render-html' }
)

log('Eval complete → ' + judgedFile)
log('HTML report → ~/html/sapphire-holistic-eval.html')

return {
  judged_file: judgedFile,
  scenario_judgments: scenarioJs.length,
  tool_judgments: toolJs.length,
  cross_model_comparisons: crossModelResults.length,
}
