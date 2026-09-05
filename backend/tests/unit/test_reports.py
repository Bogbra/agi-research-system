"""Unit tests for the pure markdown report renderer.

Regression coverage for a real readability bug: each paper's fields were
originally emitted as consecutive single-line strings, which CommonMark
collapses into one run-on paragraph (a single `\n` is not a paragraph
break). They're bullet-list items now, which render as separate lines.
"""

from __future__ import annotations

from datetime import datetime

from agiresearch.domain.schemas import (
    EvaluatedPaper,
    Paper,
    PaperEvaluation,
    PaperMetadata,
    ParameterScore,
    ParameterScores,
    ResearchState,
)
from agiresearch.domain.scoring import AGI_PARAMETERS
from agiresearch.reports import render_final_report


def _evaluated_paper(title: str, score: float) -> EvaluatedPaper:
    scores = ParameterScores(
        **{name: ParameterScore(score=score / 10, reasoning="t") for name in AGI_PARAMETERS}
    )
    evaluation = PaperEvaluation(
        parameter_scores=scores,
        overall_assessment="An assessment.",
        key_innovations=["Innovation A", "Innovation B"],
    ).with_computed_score()
    paper = Paper(
        id="1234.5678",
        title=title,
        link="https://arxiv.org/abs/1234.5678",
        metadata=PaperMetadata(
            authors=["A. Researcher"],
            abstract="An abstract long enough to pass validation elsewhere.",
            published_date=datetime(2026, 1, 1),
            categories=["cs.AI"],
        ),
    )
    return EvaluatedPaper(paper=paper, evaluation=evaluation)


def test_report_with_no_evaluated_papers_says_so():
    state = ResearchState(request_id="r1", research_objective="find agi papers")
    report = render_final_report(state)
    assert "No papers were successfully evaluated." in report


def test_each_paper_field_is_its_own_bullet_line():
    state = ResearchState(
        request_id="r1",
        research_objective="find agi papers",
        evaluated_papers=[_evaluated_paper("Paper One", 80.0)],
    )
    report = render_final_report(state)
    lines = report.splitlines()

    # Each field must start its own line — the bug this guards against
    # collapsed all of these into a single run-on paragraph.
    assert any(line.startswith("- **Authors:**") for line in lines)
    assert any(line.startswith("- **AGI score:**") for line in lines)
    assert any(line.startswith("- **Key innovations:**") for line in lines)
    assert any(line.startswith("- **Assessment:**") for line in lines)
    assert any(line.startswith("- **Link:**") for line in lines)


def test_link_is_a_markdown_autolink():
    state = ResearchState(
        request_id="r1",
        research_objective="find agi papers",
        evaluated_papers=[_evaluated_paper("Paper One", 80.0)],
    )
    report = render_final_report(state)
    assert "<https://arxiv.org/abs/1234.5678>" in report


def test_papers_are_ranked_by_score_descending():
    state = ResearchState(
        request_id="r1",
        research_objective="find agi papers",
        evaluated_papers=[
            _evaluated_paper("Low Scorer", 20.0),
            _evaluated_paper("High Scorer", 90.0),
        ],
    )
    report = render_final_report(state)
    assert report.index("High Scorer") < report.index("Low Scorer")
