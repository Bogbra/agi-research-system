# ADR 0004: Deduplication and validation are pure functions, not LLM judgment

## Context

Raw arXiv search results can contain exact and near-exact duplicates (the
same paper indexed under a resubmission or a minor revision) and entries
too sparse to evaluate meaningfully (an abstract stub). Deciding whether
two titles are "the same paper" or whether an abstract has enough content
to judge doesn't need a model — it needs a normalization rule and a
length threshold, applied consistently every time.

## Decision

`tools/arxiv_search.py` splits discovery into two functions:
`search_arxiv` (network I/O — queries arXiv, or `fake_arxiv.py`'s
generator, and maps results onto the typed `Paper` contract) and
`deduplicate_and_validate` (pure — normalizes titles for duplicate
detection, drops abstracts under `MIN_ABSTRACT_LENGTH`). Only the second
half needs a test suite, and it doesn't need arXiv, the `arxiv` package,
or a network connection to get one — `tests/unit/test_arxiv_search.py`
calls it directly against fixture `Paper` lists.

## Consequences

- Deduplication is deterministic and explainable: a paper is dropped
  because its normalized title exactly matches an earlier one, not
  because a model judged them similar. `DiscoveryStats` reports exactly
  how many were removed and why (`duplicates_removed`, `invalid_removed`),
  so a research run's paper count is always traceable to a concrete rule.
- This deliberately doesn't catch near-duplicate titles with minor wording
  differences, or two different arXiv IDs for the same paper under a
  journal cross-listing. A semantic or fuzzy-match dedup pass is real
  future work, not something this eval needs today — the exact-match rule
  catches the common case (the same paper reappearing verbatim across
  overlapping searches) without introducing a similarity threshold that
  would itself need tuning and evaluation.
