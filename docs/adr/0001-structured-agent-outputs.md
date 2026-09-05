# ADR 0001: Structured LLM outputs instead of manual JSON recovery

## Context

Two LLM calls in this pipeline need machine-readable output: the planner
(search keywords, categories, a date range) and the evaluator (ten scored
AGI parameters plus a written assessment). Parsing that out of free text by
hand — strip a ` ```json ` fence, `json.loads` it, catch the exception,
fall back to a hand-written default plan or regex-scrape individual scores
out of a broken response — means a model that wraps its JSON in a sentence,
uses smart quotes, or truncates a long response degrades silently into a
default value instead of failing loudly.

## Decision

Every LLM call binds to a Pydantic model via
`llm.with_structured_output(SomeModel)`: `agents/planner.py` binds to
`ExecutionPlan`, `agents/evaluator.py` binds to `PaperEvaluation`. A
malformed response is a `ValidationError` the caller can catch and retry —
never a value silently swapped for a default.

`ExecutionPlan` is flat and typed (`search_keywords: list[str]`,
`date_from: date`, `date_to: date`, ...) rather than a nested dict of
loosely-typed fields that has to be parsed back out by hand at each call
site.

## Consequences

- No markdown-fence stripping, no `json.loads` try/except, no regex
  fallback anywhere in this codebase. The failure mode moved from "silent
  wrong default" to "visible exception at the call site."
- **A real schema constraint this design ran into, not a hypothetical
  one**: OpenAI's strict structured-output mode rejects an open-ended
  `dict[str, Model]` field outright — `parameter_scores:
  dict[str, ParameterScore]` failed with *"'required' is required to be
  supplied and to be an array including every key in properties"*, because
  there's no fixed `properties` set for an arbitrary-key dict to satisfy
  that against. Fixed by making `ParameterScores` ten explicit named
  fields (one per entry in `domain/scoring.py:AGI_PARAMETERS`) instead of
  a dict — which also means a judge that skips a parameter is a validation
  error, not a silently incomplete dict, consistent with every other model
  in `domain/schemas.py`. The field names and `AGI_PARAMETERS`' keys must
  stay in sync; `tests/unit/test_schemas.py` asserts that directly so a
  drift fails a test, not a live judge call.
- `domain/schemas.py`'s Pydantic models are the only place either agent's
  output shape is defined — no second, informal definition of "what a
  plan looks like" living in a prompt string.
