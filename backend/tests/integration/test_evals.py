"""Integration tests for the eval harnesses — structural correctness against
the offline fake judge (forced by conftest.py), not eval *quality*, which
needs a real provider (see each eval module's own docstring).
"""

from __future__ import annotations

from agiresearch.evals.judge_reliability import (
    load_calibration_cases,
    run_calibration_eval,
    run_self_consistency_eval,
)
from agiresearch.evals.run import load_golden_papers, run_evals


def test_golden_papers_fixture_is_well_formed():
    papers = load_golden_papers()
    assert len(papers) >= 3
    for paper in papers:
        assert paper["title"]
        assert len(paper["abstract"]) >= 50
        assert paper["expected_classification"] in {
            "Low AGI Potential",
            "Medium AGI Potential",
            "High AGI Potential",
        }
        # Range labels, not an exact expected score — see evals/run.py's docstring.
        assert 0 <= paper["min_score"] <= paper["max_score"] <= 100


def test_golden_case_eval_runs_offline_and_exits_zero():
    # Fake mode is documented as non-gating (see evals/run.py) — this just
    # proves the harness runs end to end with no API key.
    assert run_evals(provider=None, model=None) == 0


def test_calibration_cases_fixture_is_well_formed():
    cases = load_calibration_cases()
    assert len(cases) >= 10  # one per named failure mode (see item 7)
    failure_modes = {case["failure_mode"] for case in cases}
    assert len(failure_modes) == len(cases)  # each failure mode covered once
    for case in cases:
        assert case["abstract"]
        assert 0 <= case["min_score"] <= case["max_score"] <= 100
        assert case["note"]


def test_calibration_eval_runs_offline():
    summary = run_calibration_eval()
    assert summary["total"] == len(load_calibration_cases())


def test_self_consistency_eval_runs_offline():
    summary = run_self_consistency_eval(repeats=2)
    assert len(summary["scores"]) == 2
    assert summary["range"] >= 0
