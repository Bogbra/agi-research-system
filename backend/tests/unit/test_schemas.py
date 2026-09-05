"""Unit tests for the typed contracts in domain/schemas.py."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from agiresearch.domain.schemas import (
    ExecutionPlan,
    Paper,
    PaperEvaluation,
    PaperMetadata,
    ParameterScore,
    ParameterScores,
)
from agiresearch.domain.scoring import AGI_PARAMETERS


def _full_parameter_scores(value: float = 5.0) -> dict[str, ParameterScore]:
    return {name: ParameterScore(score=value, reasoning="test") for name in AGI_PARAMETERS}


def test_execution_plan_rejects_inverted_date_range():
    with pytest.raises(ValidationError):
        ExecutionPlan(
            search_keywords=["agi"],
            date_from=date(2026, 1, 31),
            date_to=date(2026, 1, 1),
        )


def test_execution_plan_accepts_valid_range():
    plan = ExecutionPlan(
        search_keywords=["agi", "general intelligence"],
        date_from=date(2026, 1, 1),
        date_to=date(2026, 1, 31),
    )
    assert plan.max_papers == 10


def test_parameter_scores_requires_every_parameter():
    scores = _full_parameter_scores()
    del scores[next(iter(AGI_PARAMETERS))]
    with pytest.raises(ValidationError):
        ParameterScores(**scores)


def test_parameter_scores_rejects_unknown_parameter():
    scores = _full_parameter_scores()
    scores["not_a_real_parameter"] = ParameterScore(score=5, reasoning="bogus")
    with pytest.raises(ValidationError):
        ParameterScores(**scores)


def test_parameter_scores_field_names_match_agi_parameters():
    # Guards against the two definitions (this model's fields, and
    # AGI_PARAMETERS' keys) drifting apart silently.
    assert set(ParameterScores.model_fields) == set(AGI_PARAMETERS)


def test_paper_evaluation_computes_score_on_demand():
    evaluation = PaperEvaluation(
        parameter_scores=ParameterScores(**_full_parameter_scores(10.0)),
        overall_assessment="strong paper",
    )
    assert evaluation.agi_score is None  # not computed until asked

    computed = evaluation.with_computed_score()
    assert computed.agi_score == 100.0
    assert computed.classification == "High AGI Potential"
    assert evaluation.agi_score is None  # original untouched (model_copy)


def test_parameter_score_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ParameterScore(score=11, reasoning="too high")
    with pytest.raises(ValidationError):
        ParameterScore(score=0, reasoning="too low")


def test_paper_round_trips():
    paper = Paper(
        id="1234.5678",
        title="Example Paper",
        link="https://arxiv.org/abs/1234.5678",
        metadata=PaperMetadata(
            authors=["A. Researcher"],
            abstract="An abstract long enough to pass validation elsewhere.",
            published_date=datetime(2026, 1, 15),
            categories=["cs.AI"],
        ),
    )
    assert paper.metadata.source == "arxiv"
