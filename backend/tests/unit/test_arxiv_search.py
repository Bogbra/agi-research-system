"""Unit tests for the deterministic dedup/validation half of arXiv discovery.

No network, no `arxiv` package call — these test `deduplicate_and_validate`
directly against fixture `Paper` objects, per docs/adr/0004.
"""

from __future__ import annotations

from datetime import datetime

from agiresearch.domain.schemas import Paper, PaperMetadata
from agiresearch.tools.arxiv_search import MIN_ABSTRACT_LENGTH, deduplicate_and_validate

LONG_ABSTRACT = "x" * MIN_ABSTRACT_LENGTH
SHORT_ABSTRACT = "too short"


def _paper(id_: str, title: str, abstract: str = LONG_ABSTRACT) -> Paper:
    return Paper(
        id=id_,
        title=title,
        link=f"https://arxiv.org/abs/{id_}",
        metadata=PaperMetadata(
            authors=["A. Author"],
            abstract=abstract,
            published_date=datetime(2026, 1, 1),
            categories=["cs.AI"],
        ),
    )


def test_exact_duplicate_titles_are_deduplicated():
    papers = [_paper("1", "Same Title"), _paper("2", "Same Title")]
    result = deduplicate_and_validate(papers)
    assert result.stats.duplicates_removed == 1
    assert len(result.papers) == 1


def test_duplicate_detection_is_case_and_whitespace_insensitive():
    papers = [_paper("1", "A   Paper Title"), _paper("2", "a paper title")]
    result = deduplicate_and_validate(papers)
    assert result.stats.duplicates_removed == 1


def test_short_abstract_is_dropped_as_invalid():
    papers = [_paper("1", "Valid", LONG_ABSTRACT), _paper("2", "Invalid", SHORT_ABSTRACT)]
    result = deduplicate_and_validate(papers)
    assert result.stats.invalid_removed == 1
    assert [p.id for p in result.papers] == ["1"]


def test_stats_counts_are_internally_consistent():
    papers = [
        _paper("1", "Title A"),
        _paper("2", "Title A"),  # duplicate
        _paper("3", "Title B", SHORT_ABSTRACT),  # invalid
        _paper("4", "Title C"),
    ]
    result = deduplicate_and_validate(papers)
    stats = result.stats
    assert stats.initial_count == 4
    assert stats.after_deduplication == 3  # one dup removed
    assert stats.final_count == 2  # one invalid removed
    assert stats.duplicates_removed == 1
    assert stats.invalid_removed == 1
    assert len(result.papers) == stats.final_count


def test_empty_input_produces_empty_result():
    result = deduplicate_and_validate([])
    assert result.papers == []
    assert result.stats.initial_count == 0
    assert result.stats.final_count == 0
