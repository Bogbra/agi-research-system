# AGI Research Intelligence System

A multi-agent research pipeline: a planner turns a research objective into
search keywords and a date range, a deterministic discovery step finds and
deduplicates arXiv papers, and an evaluator agent scores each paper against
a ten-parameter, weighted "AGI-advancement potential" rubric — with the
judge itself calibrated and evaluated, not just the pipeline.

It's built as a service: a LangGraph agent pipeline behind a FastAPI API,
persisted to Postgres/SQLite, with a Next.js dashboard on top.

## Why it's built this way

The short version — the full reasoning for each decision is in
[`docs/adr/`](docs/adr/):

- **[Structured outputs, not regex-parsed prose.](docs/adr/0001-structured-agent-outputs.md)**
  Every agent's response is a validated Pydantic model. Along the way, a
  real schema constraint surfaced and got fixed: OpenAI's strict
  structured-output mode rejects an open-ended `dict[str, Model]` field
  outright, so per-parameter scores became ten explicit named fields.
- **[A direct tool call for discovery, not a ReAct agent.](docs/adr/0002-direct-tool-call-instead-of-react-agent.md)**
  By the time discovery runs, the plan has already fully determined the
  search arguments — wrapping that in an LLM tool-calling loop adds failure
  modes (skipped calls, wrong arguments, duplicate calls) with no
  corresponding benefit.
- **[The LLM provider is a config value, not a code path.](docs/adr/0003-provider-agnostic-llm-layer.md)**
  `LLM_PROVIDER=fake` and `ARXIV_PROVIDER=fake` (both defaults) run the
  entire real pipeline against deterministic offline stand-ins. No API key
  or network access is required to run this project or its test suite.
- **[Deduplication and validation are pure functions.](docs/adr/0004-deterministic-discovery-pipeline.md)**
  Deciding whether two titles are the same paper doesn't need a model — a
  normalized-title match and an abstract-length floor are deterministic,
  explainable, and unit-tested with zero network dependency.
- **[Persistence and auth, sized to what this domain actually has.](docs/adr/0005-persistence-and-auth.md)**
  One table, one bearer-token role — no fabricated audit log or reviewer
  role copied from a domain that actually needs them.
- **[The judge is evaluated, not just the pipeline.](docs/adr/0006-agi-judge-evaluation.md)**
  Golden-case classification, calibration against hype-vs-substance probes,
  and repeat-call self-consistency — run for real against `gpt-4o-mini`,
  not asserted. The honest results, including the methodology's own limits,
  are in the ADR.

## Architecture

```
                    ┌─────────────────────────────┐
                    │      Next.js dashboard       │
                    │  (Server Components +        │
                    │   Server Actions; token       │
                    │   never reaches the browser)  │
                    └───────────────┬───────────────┘
                                    │ REST (bearer auth)
                    ┌───────────────▼───────────────┐
                    │           FastAPI              │
                    │  research router · auth ·      │
                    │  rate limiting · SQLAlchemy     │
                    │  (ResearchRunRecord)            │
                    └───────────────┬───────────────┘
                                    │ BackgroundTasks
                    ┌───────────────▼───────────────┐
                    │        LangGraph pipeline      │
                    │                                │
                    │   supervisor (phase machine)   │
                    │        │      │      │         │
                    │     planner discovery evaluation│ ← sequential
                    │        │      │      │         │
                    │        └──────┴──────┘         │
                    │            supervisor          │
                    └───────────────┬───────────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
              domain/scoring   tools/arxiv_search  reports.py
              (deterministic)  (deterministic)     (deterministic)
```

## Repository layout

```
backend/            Python: domain logic, agents, graph, arXiv tool, FastAPI service
  src/agiresearch/
    domain/          Pure business logic: AGI scoring, typed schemas
    llm/             Provider-agnostic client + offline fake model
    tools/           arXiv search/dedup/validate + offline fake stand-in
    agents/          Planner, discovery ("agent" = direct tool call), evaluator
    graph/           The LangGraph pipeline definition
    api/             FastAPI app: routers, auth, persistence, rate limiting
    evals/           Golden-case classification + judge reliability/calibration evals
    reports.py       Markdown report rendering (pure)
  data/              Golden paper fixtures, judge-reliability calibration cases
  tests/             pytest: unit (domain/tools) + integration (graph, API, evals)
apps/web/            Next.js 15 dashboard (run queue, new-run form, run detail)
infra/               Dockerfiles + docker-compose (api, web, Postgres)
docs/adr/            Architecture decision records
```

## Running it

### Locally (fastest inner loop)

```bash
# Backend
cd backend
uv sync --extra dev
uv run uvicorn agiresearch.api.main:app --reload --port 8000

# Frontend, in another terminal
cd apps/web
npm install
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`, go to **New run**, describe an objective, and
submit — no API key needed, `LLM_PROVIDER=fake` / `ARXIV_PROVIDER=fake` are
the defaults. Swap in a real LLM by copying `backend/.env.example` to
`backend/.env` and setting `LLM_PROVIDER=openai` + `OPENAI_API_KEY` (arXiv
discovery needs no key — `ARXIV_PROVIDER=live` just needs network access).

### Docker Compose (Postgres-backed)

```bash
docker compose -f infra/docker-compose.yml up --build
```

Dashboard at `http://localhost:3000`, API at `http://localhost:8000`.

## Testing and evals

```bash
cd backend
uv run pytest tests -q                              # unit + integration tests, all offline
uv run python -m agiresearch.evals.run               # golden-case classification eval
uv run python -m agiresearch.evals.judge_reliability # calibration + self-consistency eval
uv run ruff check src tests                          # lint
```

`tests/integration/test_graph_fake_llm.py` runs the real LangGraph pipeline
end to end — real phase machine, real dedup/validation, real weighted-score
arithmetic — against the offline fake LLM and fake arXiv stand-in, so it
needs no API key and no network access.

`evals/run.py` and `evals/judge_reliability.py` are both explained in full,
with real `gpt-4o-mini` numbers, in
[ADR 0006](docs/adr/0006-agi-judge-evaluation.md). Short version: 3/3 golden
papers classified into the right band, both calibration probes resisted
being fooled by hype or unpersuaded by unhyped substance, and repeat-call
self-consistency stayed tight (stdev 0.76 on a 0-100 scale). Both scripts
also run fully offline as CI smoke tests — not a quality gate in that mode,
since the fake judge is a keyword heuristic, not a real judgment.

## Security posture (and its limits)

- Auth is a single bearer token; the frontend keeps it server-side and
  never ships it to the browser.
- A simple in-process rate limiter sits in front of the API — every
  research run can trigger several LLM calls, so an unmetered endpoint is
  a real cost and abuse vector even without a privileged second role to
  protect against.
- What's explicitly **not** production-hardened, and documented as such in
  [ADR 0005](docs/adr/0005-persistence-and-auth.md): the auth token is a
  placeholder for a real identity provider, there's no encryption-at-rest,
  and the rate limiter doesn't survive multiple instances.
