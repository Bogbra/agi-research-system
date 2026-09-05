"""Integration tests for fault-tolerant, per-paper evaluation in the graph.

Monkeypatches `agents.evaluator.evaluate_paper` (the single-call function
`evaluate_paper_with_retry` wraps) so one specific paper fails every
attempt while the rest succeed normally through the real
`evaluation_node` / `supervisor_node` — no stubbed LLM needed, since the
fake LLM already succeeds deterministically for anything it's given.
"""

from __future__ import annotations

from datetime import datetime

import agiresearch.agents.evaluator as evaluator_module
from agiresearch.domain.schemas import Paper, PaperMetadata, ResearchPhase, ResearchState
from agiresearch.graph.build import evaluation_node, supervisor_node


def _paper(id_: str) -> Paper:
    return Paper(
        id=id_,
        title=f"Paper {id_}",
        link=f"https://example.invalid/{id_}",
        metadata=PaperMetadata(
            authors=["A. Researcher"],
            abstract="An abstract long enough to pass validation elsewhere.",
            published_date=datetime(2026, 1, 1),
            categories=["cs.AI"],
        ),
    )


def _failing_evaluate_paper(bad_id: str, real_evaluate_paper):
    def _evaluate(paper: Paper, llm=None):
        if paper.id == bad_id:
            raise RuntimeError(f"simulated failure for {paper.id}")
        return real_evaluate_paper(paper, llm=llm)

    return _evaluate


def test_one_failed_paper_does_not_stop_remaining_evaluations(monkeypatch):
    real_evaluate_paper = evaluator_module.evaluate_paper
    monkeypatch.setattr(
        evaluator_module, "evaluate_paper", _failing_evaluate_paper("bad", real_evaluate_paper)
    )

    state = ResearchState(
        request_id="r1",
        research_objective="test",
        current_phase=ResearchPhase.EVALUATION,
        discovered_papers=[_paper("good-1"), _paper("bad"), _paper("good-2")],
    )
    update = evaluation_node(state)

    assert len(update["evaluated_papers"]) == 2
    assert {p.paper.id for p in update["evaluated_papers"]} == {"good-1", "good-2"}
    assert len(update["evaluation_failures"]) == 1


def test_failure_metadata_is_stored(monkeypatch):
    real_evaluate_paper = evaluator_module.evaluate_paper
    monkeypatch.setattr(
        evaluator_module, "evaluate_paper", _failing_evaluate_paper("bad", real_evaluate_paper)
    )

    state = ResearchState(
        request_id="r1",
        research_objective="test",
        current_phase=ResearchPhase.EVALUATION,
        discovered_papers=[_paper("bad")],
    )
    update = evaluation_node(state)

    assert len(update["evaluation_failures"]) == 1
    failure = update["evaluation_failures"][0]
    assert failure.paper_id == "bad"
    assert failure.paper_title == "Paper bad"
    assert failure.error_type == "RuntimeError"
    assert "simulated failure for bad" in failure.error_message
    assert failure.attempts == 3


def test_partial_success_completes_normally_with_failures_as_warnings(monkeypatch):
    real_evaluate_paper = evaluator_module.evaluate_paper
    monkeypatch.setattr(
        evaluator_module, "evaluate_paper", _failing_evaluate_paper("bad", real_evaluate_paper)
    )

    state = ResearchState(
        request_id="r1",
        research_objective="test",
        current_phase=ResearchPhase.EVALUATION,
        discovered_papers=[_paper("good-1"), _paper("bad")],
    )
    eval_update = evaluation_node(state)
    state = state.model_copy(update=eval_update)

    supervisor_update = supervisor_node(state)

    assert supervisor_update["current_phase"] == ResearchPhase.COMPLETION
    assert supervisor_update["final_report"] is not None
    assert "Paper good-1" in supervisor_update["final_report"]
    assert any("bad" in e for e in supervisor_update["errors"])


def test_all_papers_failing_is_a_distinct_phase_not_normal_completion(monkeypatch):
    def _always_fail(paper: Paper, llm=None):
        raise RuntimeError("everything is broken")

    monkeypatch.setattr(evaluator_module, "evaluate_paper", _always_fail)

    state = ResearchState(
        request_id="r1",
        research_objective="test",
        current_phase=ResearchPhase.EVALUATION,
        discovered_papers=[_paper("only-1"), _paper("only-2")],
    )
    eval_update = evaluation_node(state)
    state = state.model_copy(update=eval_update)

    assert eval_update["evaluated_papers"] == []
    assert len(eval_update["evaluation_failures"]) == 2

    supervisor_update = supervisor_node(state)

    assert supervisor_update["current_phase"] == ResearchPhase.EVALUATION_FAILED
    assert supervisor_update["current_phase"] != ResearchPhase.COMPLETION
    assert any("failed evaluation" in e for e in supervisor_update["errors"])
    # A report is still produced — the "no papers were successfully
    # evaluated" branch reports.py already had — but the phase, not the
    # report text, is what a consumer must check to distinguish this case.
    assert supervisor_update["final_report"] is not None


def test_final_report_can_be_produced_from_partial_results(monkeypatch):
    real_evaluate_paper = evaluator_module.evaluate_paper
    monkeypatch.setattr(
        evaluator_module, "evaluate_paper", _failing_evaluate_paper("bad", real_evaluate_paper)
    )

    state = ResearchState(
        request_id="r1",
        research_objective="test",
        current_phase=ResearchPhase.EVALUATION,
        discovered_papers=[_paper("good-1"), _paper("bad"), _paper("good-2")],
    )
    eval_update = evaluation_node(state)
    state = state.model_copy(update=eval_update)
    supervisor_update = supervisor_node(state)

    report = supervisor_update["final_report"]
    assert "Paper good-1" in report
    assert "Paper good-2" in report
    assert "Paper bad" not in report  # never evaluated, so never ranked


def test_offline_fake_mode_remains_deterministic():
    """No monkeypatching — the real evaluate_paper_with_retry against the
    real FakeChatModel, run twice, must produce identical results with
    zero failures (the fake heuristic never raises)."""

    papers = [_paper("p1"), _paper("p2"), _paper("p3")]

    def run_once() -> tuple[int, int]:
        state = ResearchState(
            request_id="r1",
            research_objective="test",
            current_phase=ResearchPhase.EVALUATION,
            discovered_papers=papers,
        )
        update = evaluation_node(state)
        return len(update["evaluated_papers"]), len(update["evaluation_failures"])

    first = run_once()
    second = run_once()

    assert first == second == (3, 0)
