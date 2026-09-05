"""The AGI research pipeline graph: a phase machine, not a fan-out/fan-in.

Sequential by nature — planning must finish before discovery has anything
to search for, and discovery must finish before there's anything to
evaluate — unlike a graph with independent branches that can run
concurrently. `supervisor_node` is a thin coordinator: not because the
phases need reconciling, but so every phase transition and its guard
condition (no plan -> stop, no papers -> stop) lives in one place instead
of scattered across edge conditions.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agiresearch.agents.evaluator import evaluate_paper_with_retry
from agiresearch.agents.planner import plan_research
from agiresearch.config import settings
from agiresearch.domain.schemas import EvaluatedPaper, Paper, ResearchPhase, ResearchState
from agiresearch.llm.client import build_chat_model
from agiresearch.reports import render_final_report
from agiresearch.services.discovery import discover_papers


def supervisor_node(state: ResearchState) -> dict:
    phase = state.current_phase

    if phase == ResearchPhase.INITIALIZATION:
        # request_id is set once, by the caller, before the graph starts
        # (see orchestrator.py) — the supervisor never generates one. A
        # second ID generated here and reconciled later was exactly the
        # bug this design replaced.
        return {"current_phase": ResearchPhase.PLANNING}

    if phase == ResearchPhase.PLANNING:
        if state.execution_plan is not None:
            return {"current_phase": ResearchPhase.DISCOVERY}
        return {
            "current_phase": ResearchPhase.COMPLETION,
            "errors": [*state.errors, "Failed to create an execution plan."],
        }

    if phase == ResearchPhase.DISCOVERY:
        if state.discovered_papers:
            return {"current_phase": ResearchPhase.EVALUATION}
        return {
            "current_phase": ResearchPhase.COMPLETION,
            "errors": [*state.errors, "No papers discovered for this objective/date range."],
        }

    if phase == ResearchPhase.EVALUATION:
        warnings = [
            f"Evaluation failed for '{f.paper_title}' ({f.paper_id}) after {f.attempts} "
            f"attempt(s): {f.error_type}: {f.error_message}"
            for f in state.evaluation_failures
        ]
        if state.evaluated_papers:
            # Some (possibly all but one) papers evaluated fine — a normal
            # completion, with per-paper failures surfaced as warnings.
            next_phase = ResearchPhase.COMPLETION
        else:
            # Every discovered paper failed evaluation — this must not
            # read like a normal completion that happened to find nothing
            # worth reporting, so it gets its own distinct phase.
            next_phase = ResearchPhase.EVALUATION_FAILED
            warnings.append(
                f"All {len(state.discovered_papers)} discovered paper(s) failed evaluation."
            )
        return {
            "current_phase": next_phase,
            "final_report": render_final_report(state),
            "errors": [*state.errors, *warnings],
        }

    return {}


def planner_node(state: ResearchState) -> dict:
    plan = plan_research(state.research_objective)
    return {"execution_plan": plan}


def discovery_node(state: ResearchState) -> dict:
    assert state.execution_plan is not None, "discovery_node requires a plan from planner_node"
    result = discover_papers(state.execution_plan)
    return {"discovered_papers": result.papers}


def evaluation_node(
    state: ResearchState,
    llm=None,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> dict:
    """Evaluate every discovered paper, isolating failures per paper and
    bounding how many run concurrently.

    One paper's evaluation failing (after `evaluate_paper_with_retry`
    exhausts its attempts) must not stop the rest from being evaluated —
    see docs/adr/0007. Papers are evaluated through a fixed-size
    `ThreadPoolExecutor` (`EVALUATION_CONCURRENCY`, default 4) rather than
    one at a time or all at once: bounded, because an unbounded burst of
    concurrent LLM calls is exactly the kind of outbound rate-limit risk a
    growing discovered-paper count would otherwise create with no cap at
    all; a pool rather than fully sequential, because paper evaluations
    are independent of each other and there's no reason to pay their
    latency one after another.

    `executor.map` (not `as_completed`) is used deliberately: it returns
    results in the same order as `state.discovered_papers` regardless of
    which worker finishes first, so a result is never ambiguously
    attributed to the wrong paper and the output is deterministic under
    the offline fake LLM even though execution order across threads is
    not. `llm` defaults to `build_chat_model()` — tests pass a stub to
    control response timing/failures without a real provider. `sleep_fn`/
    `random_fn` are forwarded to every `evaluate_paper_with_retry` call so
    tests exercising a retried failure never wait out a real backoff.
    """

    llm = llm or build_chat_model()
    evaluated: list[EvaluatedPaper] = []
    failures = list(state.evaluation_failures)

    def _evaluate(paper: Paper):
        evaluation, failure = evaluate_paper_with_retry(
            paper, llm=llm, sleep_fn=sleep_fn, random_fn=random_fn
        )
        return paper, evaluation, failure

    max_workers = max(1, settings.evaluation_concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for paper, evaluation, failure in executor.map(_evaluate, state.discovered_papers):
            if evaluation is not None:
                evaluated.append(EvaluatedPaper(paper=paper, evaluation=evaluation))
            else:
                assert failure is not None
                failures.append(failure)

    return {"evaluated_papers": evaluated, "evaluation_failures": failures}


def route_next_phase(
    state: ResearchState,
) -> Literal["planner", "discovery", "evaluation", "complete"]:
    routing: dict[ResearchPhase, Literal["planner", "discovery", "evaluation"]] = {
        ResearchPhase.PLANNING: "planner",
        ResearchPhase.DISCOVERY: "discovery",
        ResearchPhase.EVALUATION: "evaluation",
    }
    return routing.get(state.current_phase, "complete")


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("discovery", discovery_node)
    graph.add_node("evaluation", evaluation_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_next_phase,
        {
            "planner": "planner",
            "discovery": "discovery",
            "evaluation": "evaluation",
            "complete": END,
        },
    )
    for node_name in ("planner", "discovery", "evaluation"):
        graph.add_edge(node_name, "supervisor")

    return graph.compile()


@lru_cache
def get_compiled_graph():
    return build_graph()
