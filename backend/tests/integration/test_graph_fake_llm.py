"""End-to-end pipeline test against the deterministic fake LLM + fake arXiv.

Runs the real LangGraph workflow — real phase machine, real dedup/validation,
real weighted-score arithmetic — with only the LLM call and the arXiv network
call swapped for offline stand-ins. Requires no API key and no network
access, so it runs in CI on every commit.
"""

from __future__ import annotations

from agiresearch.domain.schemas import ResearchPhase
from agiresearch.orchestrator import run_research


def test_full_pipeline_runs_and_produces_a_report():
    final_state = run_research("AGI research on meta-learning from the last week")

    assert final_state.current_phase == ResearchPhase.COMPLETION
    assert final_state.execution_plan is not None
    assert final_state.discovered_papers
    assert final_state.evaluated_papers
    assert len(final_state.evaluated_papers) == len(final_state.discovered_papers)
    assert final_state.final_report is not None
    assert "AGI Research Report" in final_state.final_report
    assert not final_state.errors


def test_every_evaluated_paper_has_a_computed_score():
    final_state = run_research("survey of task transfer techniques")

    for item in final_state.evaluated_papers:
        assert item.evaluation.agi_score is not None
        assert 0 <= item.evaluation.agi_score <= 100
        assert item.evaluation.classification is not None


def test_request_id_is_assigned():
    final_state = run_research("any objective")
    assert final_state.request_id
