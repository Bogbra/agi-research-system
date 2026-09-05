"""Deterministic, offline stand-in for arXiv search — no network call.

Mirrors `llm/fake.py`'s role in the LLM layer: `ARXIV_PROVIDER=fake` (the
default) means the whole pipeline, discovery included, runs end to end with
zero network dependency. Real papers come from `ARXIV_PROVIDER=live`.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from agiresearch.domain.schemas import Paper, PaperMetadata

# Each template mentions different AGI-parameter keyword hints (see
# llm/fake.py's _PARAMETER_KEYWORDS) so a fake evaluation run downstream
# produces varied, non-identical scores instead of five identical papers.
_FAKE_ABSTRACT_TEMPLATES = [
    "A study of few-shot learning and meta-learning applied to {kw}.",
    "This work explores novel problem solving and task transfer for {kw}.",
    "We investigate abstract reasoning and world modeling in the context of {kw}.",
    "An analysis of contextual adaptation and multi-rule integration for {kw}.",
    "A survey of generalization efficiency and autonomous goal setting in {kw} systems.",
]


def generate_fake_papers(
    keywords: list[str],
    categories: list[str],
    date_from: date,
    date_to: date,
    max_papers: int,
) -> list[Paper]:
    keyword = keywords[0] if keywords else "AI"
    count = min(max_papers, len(_FAKE_ABSTRACT_TEMPLATES))
    published = datetime.combine(date_to, datetime.min.time())

    return [
        Paper(
            id=f"fake.{i:04d}",
            title=f"{keyword.title()} Research Paper {i}",
            link=f"https://example.invalid/fake-paper-{i:04d}",
            metadata=PaperMetadata(
                authors=["A. Researcher", "B. Researcher"],
                abstract=template.format(kw=keyword),
                published_date=published - timedelta(days=i),
                categories=categories or ["cs.AI"],
                source="fake",
            ),
        )
        for i, template in enumerate(_FAKE_ABSTRACT_TEMPLATES[:count], start=1)
    ]
