"""Integration tests for streaming progress (item 3) and the single
request_id source of truth (item 2).

`stream_research` is exercised directly (orchestrator level) and through
`_execute_run` (API level, with a spy on `_persist_state` to observe the
sequence of statuses written — a synchronous test can't otherwise see
"in progress" state, since BackgroundTasks-equivalent code here runs to
completion before the test can poll the DB).
"""

from __future__ import annotations

from agiresearch.api.db import ResearchRunRecord, SessionLocal, init_db
from agiresearch.api.routers import research as research_module
from agiresearch.domain.schemas import ResearchPhase, ResearchState
from agiresearch.orchestrator import run_research, stream_research


def test_stream_emits_valid_research_state_objects():
    states = list(stream_research("test objective", request_id="fixed-id"))
    assert states
    assert all(isinstance(s, ResearchState) for s in states)


def test_request_id_stays_identical_across_all_states():
    states = list(stream_research("test objective", request_id="fixed-id"))
    assert all(s.request_id == "fixed-id" for s in states)


def test_orchestrator_never_generates_a_second_id_when_one_is_given():
    states = list(stream_research("test objective", request_id="caller-supplied-id"))
    assert states[0].request_id == "caller-supplied-id"
    assert states[-1].request_id == "caller-supplied-id"


def test_run_research_generates_an_id_when_none_given():
    state = run_research("test objective")
    assert state.request_id  # non-empty, a real uuid4 string
    assert len(state.request_id) == 36  # uuid4 canonical form


def test_expected_phase_sequence_is_produced():
    phases = [s.current_phase for s in stream_research("test objective", request_id="r1")]
    # Exact phase-by-phase order matters: initialization first, completion
    # last, and each intermediate phase appears at least once, in order.
    assert phases[0] == ResearchPhase.INITIALIZATION
    assert phases[-1] == ResearchPhase.COMPLETION
    seen_order = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert seen_order == [
        ResearchPhase.INITIALIZATION,
        ResearchPhase.PLANNING,
        ResearchPhase.DISCOVERY,
        ResearchPhase.EVALUATION,
        ResearchPhase.COMPLETION,
    ]


def test_data_becomes_available_at_the_right_phase():
    states = list(stream_research("test objective", request_id="r1"))
    by_phase: dict[ResearchPhase, list[ResearchState]] = {}
    for s in states:
        by_phase.setdefault(s.current_phase, []).append(s)

    # Before planning finishes, no plan yet; the last PLANNING snapshot has one.
    assert by_phase[ResearchPhase.PLANNING][0].execution_plan is None
    assert by_phase[ResearchPhase.PLANNING][-1].execution_plan is not None

    assert by_phase[ResearchPhase.DISCOVERY][0].discovered_papers == []
    assert by_phase[ResearchPhase.DISCOVERY][-1].discovered_papers != []

    assert by_phase[ResearchPhase.EVALUATION][0].evaluated_papers == []
    assert by_phase[ResearchPhase.EVALUATION][-1].evaluated_papers != []


def test_final_state_same_meaning_as_run_research():
    streamed_final = list(stream_research("test objective", request_id="r1"))[-1]
    invoked_final = run_research("test objective", request_id="r2")

    assert streamed_final.current_phase == invoked_final.current_phase == ResearchPhase.COMPLETION
    assert len(streamed_final.evaluated_papers) == len(invoked_final.evaluated_papers)
    assert (streamed_final.final_report is not None) == (invoked_final.final_report is not None)


def test_persisted_status_follows_graph_progress(monkeypatch):
    init_db()
    request_id = "streaming-progress-test"
    db = SessionLocal()
    try:
        db.add(
            ResearchRunRecord(
                request_id=request_id,
                research_objective="test",
                status=ResearchPhase.INITIALIZATION.value,
                state_json={},
            )
        )
        db.commit()
    finally:
        db.close()

    seen_statuses: list[str] = []
    original_persist = research_module._persist_state

    def spy_persist(db, request_id, state):
        seen_statuses.append(state.current_phase.value)
        original_persist(db, request_id, state)

    monkeypatch.setattr(research_module, "_persist_state", spy_persist)

    research_module._execute_run(request_id, "test objective")

    assert seen_statuses[0] == ResearchPhase.INITIALIZATION.value
    assert ResearchPhase.PLANNING.value in seen_statuses
    assert ResearchPhase.DISCOVERY.value in seen_statuses
    assert ResearchPhase.EVALUATION.value in seen_statuses
    assert seen_statuses[-1] == ResearchPhase.COMPLETION.value

    db = SessionLocal()
    try:
        record = db.get(ResearchRunRecord, request_id)
        assert record is not None
        assert record.status == ResearchPhase.COMPLETION.value
        assert record.state_json["request_id"] == request_id
    finally:
        db.close()
