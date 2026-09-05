"""Single entry point for running one research objective through the graph."""

from __future__ import annotations

from agiresearch.domain.schemas import ResearchState
from agiresearch.graph.build import get_compiled_graph


def run_research(objective: str) -> ResearchState:
    initial_state = ResearchState(request_id="", research_objective=objective)
    graph = get_compiled_graph()
    result = graph.invoke(initial_state)
    return ResearchState.model_validate(result)
