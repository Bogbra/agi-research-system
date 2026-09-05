"""Deterministic arXiv discovery: search, deduplicate, validate.

No LLM involved — see docs/adr/0002-direct-tool-call-instead-of-react-agent.md
for why `agents/discovery.py` calls this module directly instead of
wrapping it in a ReAct tool-calling agent.

`search_arxiv` (network I/O) and `deduplicate_and_validate` (pure) are
kept separate on purpose: the pure half is what actually needs a test
suite, and it doesn't need arXiv, or even the `arxiv` package, to get one.
"""

from __future__ import annotations

import re
from datetime import date

from agiresearch.config import settings
from agiresearch.domain.schemas import DiscoveryResult, DiscoveryStats, Paper, PaperMetadata
from agiresearch.tools.fake_arxiv import generate_fake_papers

MIN_ABSTRACT_LENGTH = 50


def _build_query(keywords: list[str], categories: list[str], date_from: date, date_to: date) -> str:
    keyword_clause = " OR ".join(f'"{k}"' if " " in k else k for k in keywords)
    query = f"({keyword_clause})"
    if categories:
        cat_clause = " OR ".join(f"cat:{c}" for c in categories)
        query += f" AND ({cat_clause})"
    query += f" AND submittedDate:[{date_from:%Y%m%d}0000 TO {date_to:%Y%m%d}2359]"
    return query


def search_arxiv(
    keywords: list[str],
    categories: list[str],
    date_from: date,
    date_to: date,
    max_papers: int,
) -> list[Paper]:
    """Query arXiv and map results onto the typed `Paper` contract.

    Routes through `tools.fake_arxiv` when `ARXIV_PROVIDER=fake` (the
    default) — no network call, deterministic output, same reason
    `LLM_PROVIDER=fake` exists. Imports the `arxiv` package lazily in the
    "live" branch so `deduplicate_and_validate` (and anything testing it)
    never needs the package installed or a network connection.
    """

    if settings.arxiv_provider == "fake":
        return generate_fake_papers(keywords, categories, date_from, date_to, max_papers)

    import arxiv

    query = _build_query(keywords, categories, date_from, date_to)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    return [
        Paper(
            id=result.get_short_id(),
            title=result.title,
            link=result.entry_id,
            metadata=PaperMetadata(
                authors=[author.name for author in result.authors],
                abstract=result.summary.replace("\n", " ").strip(),
                published_date=result.published,
                categories=list(result.categories),
                source="arxiv",
            ),
        )
        for result in client.results(search)
    ]


def deduplicate_and_validate(papers: list[Paper]) -> DiscoveryResult:
    """Deduplicate by normalized title, then drop papers with a too-short
    abstract. Pure — no network, no LLM — so this is unit-tested directly
    against fixture `Paper` lists.
    """

    initial_count = len(papers)

    seen_titles: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        normalized = re.sub(r"\s+", " ", paper.title.lower().strip())
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique.append(paper)

    valid = [p for p in unique if len(p.metadata.abstract) >= MIN_ABSTRACT_LENGTH]

    stats = DiscoveryStats(
        initial_count=initial_count,
        after_deduplication=len(unique),
        final_count=len(valid),
        duplicates_removed=initial_count - len(unique),
        invalid_removed=len(unique) - len(valid),
    )
    return DiscoveryResult(papers=valid, stats=stats)


def discover_and_process_papers(
    keywords: list[str],
    categories: list[str],
    date_from: date,
    date_to: date,
    max_papers: int,
) -> DiscoveryResult:
    """Full discovery workflow: search, then deduplicate and validate."""

    papers = search_arxiv(keywords, categories, date_from, date_to, max_papers)
    return deduplicate_and_validate(papers)
