"""Unit tests for the deterministic AGI-score calculation — no LLM involved."""

from __future__ import annotations

import pytest

from agiresearch.domain.scoring import (
    AGI_PARAMETERS,
    HIGH_POTENTIAL_THRESHOLD,
    MEDIUM_POTENTIAL_THRESHOLD,
    calculate_agi_score,
)


def _all_scores(value: float) -> dict[str, float]:
    return {name: value for name in AGI_PARAMETERS}


def test_all_tens_scores_100():
    breakdown = calculate_agi_score(_all_scores(10.0))
    assert breakdown.final_score == 100.0
    assert breakdown.classification == "High AGI Potential"


def test_all_ones_scores_ten():
    breakdown = calculate_agi_score(_all_scores(1.0))
    assert breakdown.final_score == 10.0
    assert breakdown.classification == "Low AGI Potential"


def test_high_potential_boundary_is_inclusive():
    # Constructed to land exactly on HIGH_POTENTIAL_THRESHOLD.
    scores = _all_scores(HIGH_POTENTIAL_THRESHOLD / 10)
    breakdown = calculate_agi_score(scores)
    assert breakdown.final_score == HIGH_POTENTIAL_THRESHOLD
    assert breakdown.classification == "High AGI Potential"


def test_medium_potential_boundary_is_inclusive():
    scores = _all_scores(MEDIUM_POTENTIAL_THRESHOLD / 10)
    breakdown = calculate_agi_score(scores)
    assert breakdown.final_score == MEDIUM_POTENTIAL_THRESHOLD
    assert breakdown.classification == "Medium AGI Potential"


def test_well_below_medium_threshold_is_low():
    scores = _all_scores((MEDIUM_POTENTIAL_THRESHOLD - 1.0) / 10)
    breakdown = calculate_agi_score(scores)
    assert breakdown.classification == "Low AGI Potential"


def test_missing_parameter_raises_key_error():
    scores = _all_scores(5.0)
    del scores[next(iter(AGI_PARAMETERS))]
    with pytest.raises(KeyError):
        calculate_agi_score(scores)


def test_contributions_sum_to_final_score_over_ten():
    breakdown = calculate_agi_score(_all_scores(7.0))
    assert round(sum(breakdown.contributions.values()) * 10, 1) == breakdown.final_score
