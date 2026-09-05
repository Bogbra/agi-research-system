# ADR 0005: Persistence and auth, sized to what this domain actually has

## Context

A research run needs to survive the request that started it — the
pipeline runs as a FastAPI `BackgroundTask`, and a client needs to poll for
status and eventually read the report. That's the entire persistence
requirement: there is no approval workflow, no second party who reviews a
run before it's final, and no regulatory reason to keep an immutable
append-only log of who did what. Building those anyway, by habit or by
copying a pattern from a domain that does need them, would be a fabricated
requirement dressed up as thoroughness.

## Decision

- **One table.** `ResearchRunRecord` (`api/db.py`): `request_id`,
  objective, status (a `ResearchPhase` value, or `"failed"`), paper count,
  average score, the final report text, the full `ResearchState` as JSON,
  and an error field. SQLite by default, `DATABASE_URL` swaps to Postgres
  for anything shared. No second audit-log table — there's no
  human-in-the-loop action to audit.
- **One role.** `api/auth.py` checks a single static bearer token
  (`API_BEARER_TOKEN`). No `caller`/`reviewer` split — there is exactly
  one kind of caller and exactly one kind of action (submit or read a
  run), so a role split would be a distinction with nothing on the other
  side of it. Swapping this for a real identity provider touches only
  this module: every router depends on `Principal`, never a raw token.
- **A rate limiter anyway.** Every research run can trigger several LLM
  calls (one plan, one evaluation per discovered paper), so an unmetered
  endpoint is a real cost and abuse vector even without a second role to
  protect against. The same in-process token-bucket limiter as any other
  endpoint, not a distributed one — the right tool once this actually
  needs to scale horizontally, not before.

## Consequences

- `POST /research` returns `202 Accepted` immediately and hands the
  objective to a `BackgroundTasks`-run pipeline; `GET /research/{id}`
  polls the one table for status, exactly like the dashboard's
  `AutoRefresh` component expects. As of ADR 0007, that table is updated
  after every graph step, not just once at the end, so polling reflects
  real phase progress.
- If a future version of this project added a step a second party needs
  to approve or dispute, the audit-log table and the role split from a
  compliance-heavy sibling project are the known pattern to reach for —
  deliberately not built now, because nothing here needs them yet.
- **`BackgroundTasks` is not a durable job queue, and this is a real limit,
  not an oversight.** Job execution is tied to the API process: an API
  process crash or restart loses whatever run was in progress, and there
  is no persistent queue to pick it back up. This is intentionally
  sufficient for the current deployment model (a single API instance, a
  portfolio/demo context) — a deployment that needs guaranteed execution
  across process restarts would move jobs to a persistent queue/worker
  architecture (e.g. a database-backed queue, or Celery/RQ against Redis)
  instead. Not needed for this version; the cost of adding one now, before
  anything demonstrates it's actually necessary, would outweigh the
  benefit.
