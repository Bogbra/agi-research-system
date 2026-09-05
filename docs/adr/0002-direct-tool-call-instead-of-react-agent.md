# ADR 0002: A direct tool call for discovery, not a ReAct agent

## Context

By the time discovery runs, the planner has already produced a fully
specified `ExecutionPlan`: keywords, categories, a date range, a paper
limit. Wrapping the arXiv search tool in an LLM-driven ReAct agent (a model
that reads the plan, decides to call the tool, and picks its arguments)
adds a decision point where none exists — every argument the tool needs is
already sitting in a typed object. The only things an agent can add here
are new failure modes: skip the call, call it with the wrong arguments,
call it more than once, or paraphrase the plan's dates instead of passing
them through exactly.

## Decision

`agents/discovery.py` is a five-line function that reads `ExecutionPlan`'s
fields and calls `tools.arxiv_search.discover_and_process_papers(...)`
directly. No LLM, no tool-calling loop, no prompt.

## Consequences

- One fewer LLM call per research run, and one fewer place a run can fail
  non-deterministically. The discovery step is exactly as reliable as the
  arXiv API and the deterministic dedup/validation logic behind it (see
  ADR 0004) — nothing else can go wrong here.
- `graph/build.py`'s `discovery_node` is trivially unit-testable: call it
  with a `ResearchState` that has an `execution_plan` set, assert the
  right arguments reached the tool. No agent transcript to parse, no
  tool-call assertion library needed.
- This is a narrow, deliberate exception, not a rule that "agents are
  wasteful." The evaluator (`agents/evaluator.py`) *is* an LLM call,
  because judging a paper's AGI-advancement potential is exactly the kind
  of task with no deterministic implementation — see ADR 0006. The
  distinction is whether the LLM's job is answering a question with
  actual uncertainty, or just gluing together arguments that were already
  fully determined.
