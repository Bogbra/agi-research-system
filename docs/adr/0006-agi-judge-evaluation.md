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
`data/evals/judge_reliability_cases.json` holds two adversarial probes
built directly from the evaluator's own system-prompt instruction
("focus on what the paper actually demonstrates, not just claims"): one
abstract with AGI hype language ("groundbreaking", "true AGI") wrapped
around a narrow image-classification method (should score no higher than
Medium), and one with zero AGI branding describing a genuinely broad
cross-task result (should score at least Medium). Bounds, not exact
matches — the point is "not fooled in either direction," not "reproduces
one specific score." Separately, the same fixed paper is judged three
times in a row to measure the AGI-score range and standard deviation
across repeat calls — a judge whose score swings between bands on
identical input isn't a usable signal regardless of whether any one run
looks right.

## Consequences — what the eval actually found

Run against real embeddings-free `gpt-4o-mini` (`--provider openai`) on
2026-09-05:

- **Golden cases: 3/3 passed.** Low scored 26.4/100, Medium scored
  50.6/100, High scored 77.1/100 — correctly ordered and correctly banded
  on the first real run, no fixture tuning needed to get there.
- **Calibration: 2/2 passed.** The hype-language paper scored 21.6/100
  (Low) despite its "groundbreaking... true AGI" framing — the judge
  scored the narrow CIFAR-10 classifier it actually described, not its own
  marketing copy. The unhyped substantive-transfer paper scored 66.4/100
  (Medium) with zero AGI vocabulary in its abstract.
- **Self-consistency: scores `[63.8, 64.0, 62.3]`, stdev 0.76.** Tight
  enough that a single run's score is a reasonable signal on its own; this
  is worth re-checking if the model or prompt changes, not assumed to
  hold forever.
- These results are better-behaved than a first real run of an LLM-judge
  eval necessarily should be assumed to be — the honest finding here is
  "no reliability problem surfaced yet," not proof one doesn't exist
  under different phrasing, domains, or model versions. The calibration
  set is deliberately small and adversarial by construction (probing one
  specific failure pattern), not a statistically representative sample —
  see `judge_reliability.py`'s own docstring.
- Both eval scripts run fully offline as CI smoke tests (`LLM_PROVIDER=fake`,
  exit 0 regardless of classification match) and are otherwise a
  manual/periodic check before shipping a prompt change, the same reason
  the golden-case and calibration numbers above aren't a per-commit CI
  gate: they cost real API calls and, per the self-consistency check
  itself, aren't perfectly deterministic run-to-run.
