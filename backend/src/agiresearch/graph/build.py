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

import uuid
from functools import lru_cache
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agiresearch.agents.discovery import discover_papers
from agiresearch.agents.evaluator import evaluate_paper
from agiresearch.agents.planner import plan_research
from agiresearch.domain.schemas import EvaluatedPaper, ResearchPhase, ResearchState
from agiresearch.llm.client import build_chat_model
from agiresearch.reports import render_final_report


def supervisor_node(state: ResearchState) -> dict:
    phase = state.current_phase

    if phase == ResearchPhase.INITIALIZATION:
        return {
            "request_id": state.request_id or str(uuid.uuid4()),
            "current_phase": ResearchPhase.PLANNING,
        }

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
        return {
            "current_phase": ResearchPhase.COMPLETION,
            "final_report": render_final_report(state),
        }

    return {}


def planner_node(state: ResearchState) -> dict:
    plan = plan_research(state.research_objective)
    return {"execution_plan": plan}


def discovery_node(state: ResearchState) -> dict:
    assert state.execution_plan is not None, "discovery_node requires a plan from planner_node"
    result = discover_papers(state.execution_plan)
    return {"discovered_papers": result.papers}


def evaluation_node(state: ResearchState) -> dict:
    llm = build_chat_model()
    evaluated = [
        EvaluatedPaper(paper=paper, evaluation=evaluate_paper(paper, llm=llm))
        for paper in state.discovered_papers
    ]
    return {"evaluated_papers": evaluated}


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
