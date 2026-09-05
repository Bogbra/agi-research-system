# ADR 0006: Evaluating the AGI judge itself — golden cases, calibration, and self-consistency

## Context

The core of this pipeline is a single LLM call that reads a paper's
abstract and produces a judgment: ten 1-10 parameter scores, an overall
assessment, key innovations, limitations. Nothing about "does the graph
run end to end" (a fake-mode smoke test) says anything about whether that
judgment is any good. Three separate questions need separate answers:
does the judge classify obviously low/medium/high-potential papers into
the right band; can the judge be fooled by a hyped-up abstract describing
a narrow method, or undersell a substantive result that never uses AGI
buzzwords; and does the same paper get roughly the same score twice.

## Decision

**`evals/run.py` — golden-case classification.**
`data/test_cases/golden_papers.json` holds three hand-written papers
spanning the three classification bands: a narrow single-benchmark image
classifier (Low), a bounded cross-task transfer result within one domain
(Medium), and a few-shot, cross-domain, meta-learning agent with results
backing every claim (High). The eval asserts the real judge's
classification matches. Unlike a domain where the offline fake heuristic
can meaningfully reproduce the expected result (a structured,
deterministic tool output like a credit-score tier), this judge reads free
text — there's no deterministic signal for an offline heuristic to hang a
meaningful pass/fail on, so fake-mode runs are explicit non-gating smoke
tests, not a quality signal.

**`evals/judge_reliability.py` — calibration and self-consistency.**
`data/evals/judge_reliability_cases.json` holds ten hand-written probes,
one per named failure mode: hype language over a narrow method, broad
substance without AGI branding, a strong result confined to one benchmark
with no transfer, genuine cross-domain transfer, few-shot adaptation, a
pure scaling result with no new capability, a reasoning trick validated on
a single benchmark, sweeping claims backed by thin evidence, a
substantive result undersold by conservative wording, and a deliberately
ambiguous borderline case. Each is labeled with a `[min_score, max_score]`
band, not an exact value — testing a probabilistic judge against one
precise number tests noise, not calibration. Separately, the same fixed
paper is judged three times in a row to measure the AGI-score range and
standard deviation across repeat calls — a judge whose score swings
between bands on identical input isn't a usable signal regardless of
whether any one run looks right.

## Consequences — what the eval actually found

**A real-model eval result is a snapshot of one run, not a permanent
property of this system.** Every number below is tied to the exact
snapshot file it came from, under `backend/eval-results/` — each file
records `provider`, `model`, `prompt_sha256`
(`agents/evaluator.py:PROMPT_SHA256`) and `rubric_version`
(`domain/scoring.py:RUBRIC_VERSION`) precisely so that if a later run's
numbers differ, it's possible to tell whether the model, the prompt, or
the rubric changed, rather than treating a changed number as unexplained
drift (or worse, leaving a stale number in this ADR after any of those
things change). When re-running these evals, add a new snapshot file and
update the references below to it — don't edit the numbers in place
without pointing at what produced them.

Referenced snapshots (`gpt-4o-mini`, `provider=openai`,
`prompt_sha256=9d1218e6…052`, `rubric_version=1.0.0`):

- **Golden cases — [`eval-results/golden-case-20260905T194108Z.json`](../../backend/eval-results/golden-case-20260905T194108Z.json): 3/3 passed.**
  Low scored 26.4/100 (band 0-39), Medium scored 50.6/100 (band 40-69),
  High scored 77.1/100 (band 70-100) — correctly ordered and correctly
  banded, no fixture tuning needed to get there.
- **Calibration — [`eval-results/judge-reliability-20260905T194210Z.json`](../../backend/eval-results/judge-reliability-20260905T194210Z.json): 8/10 passed.**
  Two calibration misses were observed in the referenced real-model
  evaluation, reported as found rather than adjusted after the fact:
  - `reasoning-improvement-limited-to-one-benchmark` scored 40.9, one point
    above its expected [10, 40] band — a near-miss at the boundary,
    plausibly a case where the expected band was drawn slightly too tight
    rather than a real judge error.
  - `strong-claims-with-weak-evidence` scored 41.4 (Medium), well above its
    expected [0, 35] band. The abstract claims "unprecedented general
    reasoning capability applicable to any domain" backed by one small
    synthetic dataset with no baselines or ablations — the judge gave this
    meaningfully more credit than the (thin) evidence supports, the exact
    failure mode this probe was built to catch. In this run, at least,
    it's a genuine calibration gap, not a fixture-tuning artifact: the
    fixture was not adjusted after seeing this result.
  - Every other probe — including the hype-language case, the
    conservative-wording case, and the deliberately ambiguous
    borderline case — landed inside its expected band in this run.
- **Self-consistency (same snapshot file, `self_consistency` key): scores
  `[62.3, 62.3, 63.8]`, stdev 0.71.** Tight enough that this single run's
  score looks like a reasonable signal on its own; this is worth
  re-checking whenever the model or prompt changes, not assumed to hold
  indefinitely from one measurement.
- The honest summary, scoped to this one referenced snapshot: the judge
  scored substance over vocabulary correctly on 8 of 10 probes, but
  measurably under-penalized strong claims backed by weak evidence on the
  ninth — a real, useful finding rather than a clean pass, and exactly
  what this eval exists to surface. Ten cases is still not a
  statistically representative sample — see `judge_reliability.py`'s own
  docstring — and a single run is not a claim that these exact two
  probes will miss again; a second dated snapshot, added the same way as
  this one, would be needed to say anything about repeatability.
- Both eval scripts run fully offline as CI smoke tests (`LLM_PROVIDER=fake`,
  exit 0 regardless of score match, and never write to `eval-results/` in
  that mode — see `result_logging.py`) and are otherwise a manual/periodic
  check before shipping a prompt change, the same reason the golden-case
  and calibration numbers above aren't a per-commit CI gate: they cost
  real API calls and, per the self-consistency check itself, aren't
  perfectly deterministic run-to-run.
