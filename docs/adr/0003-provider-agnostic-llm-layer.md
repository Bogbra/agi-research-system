# ADR 0003: Provider-agnostic LLM layer, offline by default

## Context

Two external dependencies sit on the critical path of every research run:
an LLM (for planning and evaluation) and arXiv (for discovery). Neither
should be required to run the test suite, review this project, or run a
demo — an API key requirement or a flaky network call at review time is a
worse first impression than a clearly-labeled offline stand-in.

## Decision

Both dependencies are swappable via config, defaulting to an offline mode:

- `LLM_PROVIDER` (`fake` default, `openai`, `anthropic`) — every agent
  calls `llm.client.build_chat_model()`, never `ChatOpenAI(...)` directly.
  `llm/fake.py`'s `FakeChatModel` implements `with_structured_output` with
  a deterministic keyword heuristic per schema (`ExecutionPlan`,
  `PaperEvaluation`) — enough to exercise the full graph and every test,
  explicitly not enough to be mistaken for a real judgment (see ADR 0006
  and each eval script's own printed warning in fake mode).
- `ARXIV_PROVIDER` (`fake` default, `live`) — `tools/arxiv_search.py`
  routes to `tools/fake_arxiv.py`'s deterministic paper generator instead
  of the real arXiv API. The same reasoning as the LLM side: a graph run
  or a test shouldn't depend on network reachability by default.

`tests/conftest.py` force-sets both to `fake` at collection time (not just
defaults) — pydantic-settings resolves a real `.env` file's values before
falling back to a field default, and a developer's local `.env`
legitimately sets `LLM_PROVIDER=openai` to run a live eval. Without the
override, the test suite would silently start making real, billed API
calls the moment that `.env` file exists.

## Consequences

- `docker compose up`, `pytest`, and every eval script's default invocation
  run end to end with zero external dependencies and zero cost.
- Swapping to a real provider is a one-line env change, not a code change,
  at every call site.
- The fake heuristics are honest about their own limits: `llm/fake.py` and
  every eval script that runs against `LLM_PROVIDER=fake` print a warning
  that the numbers aren't meaningful, rather than let a fake-mode result
  be mistaken for a real judgment quality signal.
