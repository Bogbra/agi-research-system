"""Integration tests for bounded, ordered concurrent paper evaluation.

Uses a stub LLM (same shape as `test_evaluator_retry.py`'s) that tracks how
many calls are in flight at once, so "no unbounded parallel requests" is
verified directly rather than inferred from reading the code.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from agiresearch.config import settings
from agiresearch.domain.schemas import (
    Paper,
    PaperEvaluation,
    PaperMetadata,
    ParameterScore,
    ParameterScores,
    ResearchPhase,
    ResearchState,
)
from agiresearch.domain.scoring import AGI_PARAMETERS
from agiresearch.graph.build import evaluation_node


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


def _valid_evaluation() -> PaperEvaluation:
    scores = ParameterScores(
        **{name: ParameterScore(score=5.0, reasoning="ok") for name in AGI_PARAMETERS}
    )
    return PaperEvaluation(parameter_scores=scores, overall_assessment="fine")


class _ConcurrencyTrackingLLM:
    """Every `.invoke()` sleeps briefly and records the peak number of
    calls that were in flight simultaneously — a real assertion that
    concurrency stayed bounded, not just an assumption from reading the
    executor's `max_workers`."""

    def __init__(self, delay_seconds: float = 0.05) -> None:
        self._delay = delay_seconds
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.call_count = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.call_count += 1
        time.sleep(self._delay)
        with self._lock:
            self._in_flight -= 1
        return _valid_evaluation()


def _state_with_papers(count: int) -> ResearchState:
    return ResearchState(
        request_id="r1",
        research_objective="test",
        current_phase=ResearchPhase.EVALUATION,
        discovered_papers=[_paper(f"p{i}") for i in range(count)],
    )


def test_concurrency_never_exceeds_configured_limit(monkeypatch):
    monkeypatch.setattr(settings, "evaluation_concurrency", 2)
    llm = _ConcurrencyTrackingLLM(delay_seconds=0.05)

    state = _state_with_papers(8)
    update = evaluation_node(state, llm=llm)

    assert len(update["evaluated_papers"]) == 8
    assert llm.max_in_flight <= 2
    # With 8 papers, a real limit of 1 would serialize everything — this
    # confirms the pool is actually running work in parallel, not just
    # capping an already-sequential loop.
    assert llm.max_in_flight >= 2


def test_concurrency_of_one_is_fully_sequential(monkeypatch):
    monkeypatch.setattr(settings, "evaluation_concurrency", 1)
    llm = _ConcurrencyTrackingLLM(delay_seconds=0.02)

    state = _state_with_papers(5)
    evaluation_node(state, llm=llm)

    assert llm.max_in_flight == 1


def test_results_map_unambiguously_to_their_paper_regardless_of_finish_order():
    """Papers with different simulated latencies finish out of order under
    the executor; the returned list must still be in the same order as
    `discovered_papers`, and each evaluated entry must be *that* paper."""

    class _VariableDelayLLM:
        def __init__(self):
            self._lock = threading.Lock()

        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            text = "\n".join(str(m.content) for m in messages)
            # The slowest paper is listed first, so if order were determined
            # by completion time rather than input order, it would land last.
            if "TITLE:\nPaper p0" in text:
                time.sleep(0.08)
            else:
                time.sleep(0.01)
            return _valid_evaluation()

    papers = [_paper(f"p{i}") for i in range(4)]
    state = ResearchState(
        request_id="r1",
        research_objective="test",
        current_phase=ResearchPhase.EVALUATION,
        discovered_papers=papers,
    )
    update = evaluation_node(state, llm=_VariableDelayLLM())

    assert [ep.paper.id for ep in update["evaluated_papers"]] == ["p0", "p1", "p2", "p3"]


def test_offline_fake_mode_is_deterministic_under_concurrency():
    state = _state_with_papers(6)

    first = evaluation_node(state)
    second = evaluation_node(state)

    first_scores = [ep.evaluation.agi_score for ep in first["evaluated_papers"]]
    second_scores = [ep.evaluation.agi_score for ep in second["evaluated_papers"]]
    assert first_scores == second_scores
    assert len(first["evaluation_failures"]) == len(second["evaluation_failures"]) == 0
