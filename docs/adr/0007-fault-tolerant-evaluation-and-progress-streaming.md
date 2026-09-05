# ADR 0007: Fault-tolerant, bounded-concurrency evaluation, one request_id, and real progress

## Context

Three related reliability gaps surfaced once this pipeline was exercised
end to end rather than just unit-tested phase by phase:

1. `evaluation_node` evaluated every discovered paper in a single list
   comprehension. One paper triggering a transient provider error or a
   malformed structured-output response took the entire research run down
   with it — nine successfully-evaluatable papers lost because the tenth
   hit a rate limit.
2. Both the API layer and the graph could mint a `request_id`: the API
   generated one at submission time, the graph's supervisor generated a
   second one if it didn't see one already set, and `_execute_run` then
   overwrote the graph's id with the API's id after the fact via
   `model_copy(update={"request_id": ...})`. Two sources of truth
   reconciled by a workaround is itself a bug waiting to happen.
3. The database record was written exactly once, after the entire graph
   finished. A client polling `GET /research/{id}` while a run was in
   progress saw the same `initialization` status over and over, then a
   single jump straight to `completion` — the phase machine existed in
   the state model but was invisible to anyone polling it.

## Decision

**Fault-tolerant evaluation.** `agents/evaluator.py:evaluate_paper_with_retry`
wraps `evaluate_paper` with up to `MAX_EVALUATION_ATTEMPTS` (3) retries,
waiting between attempts with exponential backoff plus jitter
(`_backoff_seconds`: ~0.5s before attempt 2, ~1.0s before attempt 3, each
plus up to 0.25s of random jitter) — enough to ride out a brief provider
hiccup or rate-limit window, still small since there are at most two
waits. Only `_is_retryable` failures are retried at all: a malformed
structured-output response (`pydantic.ValidationError`) and known-transient
provider errors (connection/timeout/rate-limit/5xx, from either the
`openai` or `anthropic` SDK). Everything else — an authentication error, a
bad request, a plain `TypeError` from a bug in this code — fails on the
first attempt: retrying an exact repeat of a call that structurally can't
succeed only burns attempts (and, against a real provider, money) for no
chance of a different outcome. `EvaluationFailure.attempts` reflects how
many tries actually happened, so a fail-fast case is visibly `attempts=1`,
not padded to look like retries ran. It never raises: it returns
`(evaluation, None)` on success or `(None, EvaluationFailure)` once
retries are exhausted or a non-retryable error is hit.
`graph/build.py:evaluation_node` calls this once per paper (through the
thread pool described below), appending to either `evaluated_papers` or
`evaluation_failures` — one paper's exhausted retries never stop the loop.
`sleep_fn`/`random_fn` are injectable (default `time.sleep`/`random.random`)
and threaded through `evaluation_node` too, so tests exercising a retried
failure run in milliseconds instead of actually waiting out the backoff.

`supervisor_node`'s `EVALUATION` branch then distinguishes two outcomes
explicitly:
- **At least one paper evaluated successfully** → `ResearchPhase.COMPLETION`,
  with every failure recorded in `errors` as a human-readable warning
  alongside the (partial) report.
- **Every discovered paper failed** → `ResearchPhase.EVALUATION_FAILED`, a
  distinct phase value specifically so this can never be mistaken, in the
  API response or the dashboard, for a normal completion that happened to
  find nothing worth reporting.

**One `request_id`.** `orchestrator.run_research(objective, request_id=None)`
and the new `stream_research` (below) generate a `request_id` exactly once,
before the graph starts, if the caller didn't supply one. The graph's
`supervisor_node` no longer generates or touches `request_id` at all. The
API's `submit_research` generates the id and passes it straight through;
`_execute_run`'s `model_copy(update={"request_id": ...})` workaround is
gone because there's nothing left to reconcile.

**Real progress.** `orchestrator.stream_research(objective, request_id=None)`
is `run_research`'s streaming twin: same graph, same single `request_id`,
but it yields a `ResearchState` snapshot after every graph step
(`graph.stream(..., stream_mode="values")`) instead of only the final one.
`api/routers/research.py:_execute_run` iterates this and persists a
snapshot to `ResearchRunRecord` after each one, so `status` in the database
— and therefore what a polling client sees — actually tracks
`initialization → planning → discovery → evaluation → completion` (or
`evaluation_failed`) as the graph runs, not just at the very end. Data
becomes visible at the phase where it's produced: `execution_plan` appears
once planning's snapshot includes it, `discovered_papers` once discovery's
does, `evaluated_papers` once evaluation's does.

**Bounded concurrency.** Once the fault-tolerance above made a run
robust to any single paper's failure, `evaluation_node` was still
evaluating papers one at a time — with real network latency per LLM call,
the whole run's latency scaled linearly with paper count for no reason
other than that nothing had parallelized it yet. Papers now run through a
fixed-size `ThreadPoolExecutor` (`EVALUATION_CONCURRENCY`, default 4) via
`executor.map`, not `as_completed`: `map` returns results in the same
order as `discovered_papers` regardless of which worker finishes first, so
a result is never ambiguously attributed to the wrong paper even though
completion order across threads isn't deterministic. The limit is a hard
cap, not a target — 10 discovered papers with `EVALUATION_CONCURRENCY=4`
run in three batches (4, 4, 2), never 10 requests in flight at once.

## Consequences

- A research run's reliability no longer degrades to "all or nothing" as
  paper count grows — the more papers discovered, the more surface area
  for one to hit a transient failure, and that no longer means losing the
  whole run's findings.
- `EvaluationFailure` (paper id, title, error type, error message, attempt
  count) is a first-class part of `ResearchState` and the `RunDetail` API
  response — a caller can distinguish "this paper's evaluation failed
  three times, here's why" from "this paper was silently dropped."
- Retry classification means a real, reproducible bug in this codebase
  fails a paper's evaluation in one attempt, not three — `attempts=1` in
  the resulting `EvaluationFailure` is itself evidence of which category a
  failure fell into, without reading logs.
- Backoff timing (`_backoff_seconds`) and the retryable/non-retryable
  split (`_is_retryable`) are each covered by dedicated unit tests
  (`tests/unit/test_evaluator_retry.py`) using injected `sleep_fn`/
  `random_fn` — none of them, or the graph-level fault-tolerance tests
  that exercise a fully-retried failure, wait out a real delay.
- **Measured, not assumed, speedup**: the same 5-paper research run took
  22.1s wall-clock at `EVALUATION_CONCURRENCY=1` (fully sequential) and
  11.7s at the default of 4 — roughly 1.9x for 5 papers bounded by a limit
  of 4, run against real `gpt-4o-mini` on 2026-09-05. The isolation from
  the fault-tolerance work above meant this was a pure throughput change:
  the same 0 failures, same evaluated count, in about half the time.
- `tests/integration/test_evaluation_concurrency.py` verifies the bound
  directly (a tracking stub LLM records peak concurrent in-flight calls,
  asserted `<=` the configured limit, not inferred from reading
  `max_workers`) and that `evaluation_node`'s output order matches
  `discovered_papers`' order even when a deliberately slower paper is
  listed first and finishes last.
- `stream_research` and `run_research` share the same graph and must
  produce the same final state for the same input — tested directly
  (`tests/integration/test_streaming_progress.py`) rather than assumed.
- The dashboard's existing polling (`AutoRefresh`) needed no change — it
  already re-fetched `GET /research/{id}` on an interval; it was the
  *data behind* that endpoint that was static mid-run, not the polling
  itself.
