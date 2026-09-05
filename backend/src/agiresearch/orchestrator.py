"""Entry points for running one research objective through the graph.

`run_research` is the simple, synchronous interface: invoke the graph, get
the final state back. `stream_research` is the same run, but yields a
`ResearchState` snapshot after every graph step, so a caller (the API
layer) can persist real progress instead of only learning the outcome once
the whole run has finished. See docs/adr/0007.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from agiresearch.domain.schemas import ResearchState
from agiresearch.graph.build import get_compiled_graph


def _initial_state(objective: str, request_id: str | None) -> ResearchState:
    return ResearchState(request_id=request_id or str(uuid.uuid4()), research_objective=objective)


def run_research(objective: str, request_id: str | None = None) -> ResearchState:
    """Run `objective` to completion and return the final state.

    `request_id` is generated here (a fresh uuid4) if not supplied — the
    graph itself never generates one, so the id a caller passes in is
    guaranteed to be the same id on every state this run ever produces.
    """

    initial_state = _initial_state(objective, request_id)
    graph = get_compiled_graph()
    result = graph.invoke(initial_state)
    return ResearchState.model_validate(result)


def stream_research(objective: str, request_id: str | None = None) -> Iterator[ResearchState]:
    """Like `run_research`, but yields a `ResearchState` after every step
    instead of only returning the final one.

    Intermediate snapshots repeat the same `request_id` throughout — never
    regenerated mid-stream — so a consumer persisting each snapshot (see
    `api/routers/research.py`) never has an id mismatch between what it
    started tracking and what a later snapshot reports.
    """

    initial_state = _initial_state(objective, request_id)
    graph = get_compiled_graph()
    for update in graph.stream(initial_state, stream_mode="values"):
        yield ResearchState.model_validate(update)
