"""Evaluator agent: judges one paper's AGI-advancement potential.

Replaces a three-tier manual JSON recovery (markdown-fence strip, then
trailing-comma repair, then regex-scraping individual scores out of broken
JSON) with a single `with_structured_output(PaperEvaluation)` call. See
docs/adr/0001-structured-agent-outputs.md.

The weighted 0-100 score is computed afterward by
`PaperEvaluation.with_computed_score()` (pure arithmetic, `domain/scoring.py`)
rather than asked of the model — a judge asked to also do the weighted-sum
arithmetic is a judge that can get the arithmetic wrong on top of whatever
else it gets wrong. See docs/adr/0006-agi-judge-evaluation.md.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agiresearch.domain.schemas import EvaluationFailure, Paper, PaperEvaluation
from agiresearch.domain.scoring import AGI_PARAMETERS
from agiresearch.llm.client import build_chat_model

logger = logging.getLogger(__name__)

MAX_EVALUATION_ATTEMPTS = 3

SYSTEM_PROMPT = """You are an expert AGI evaluator. Analyze research papers for \
their contribution to Artificial General Intelligence.

Rate each of the 10 AGI parameters on a 1-10 scale:
- 1-3: no/minimal AGI relevance
- 4-6: some AGI potential but limited
- 7-8: strong AGI contribution
- 9-10: exceptional AGI breakthrough

Be conservative — reserve 7+ for truly exceptional work. Focus on what the
paper actually demonstrates, not just claims. Distinguish narrow-AI
improvements from genuine AGI progress."""


def _parameter_list() -> str:
    return "\n".join(
        f"- {name} ({weight.weight:.0%}): {weight.description}"
        for name, weight in AGI_PARAMETERS.items()
    )


def evaluate_paper(paper: Paper, llm=None) -> PaperEvaluation:
    """Judge `paper` against the 10-parameter AGI rubric and compute its
    weighted score.

    `llm` defaults to `build_chat_model()` — pass one explicitly to reuse a
    single instance across many calls instead of hitting the (cached)
    factory each time.
    """

    authors = ", ".join(paper.metadata.authors[:5])
    user_prompt = f"""TITLE:
{paper.title}

AUTHORS:
{authors}

ABSTRACT:
{paper.metadata.abstract}

AGI PARAMETERS TO SCORE:
{_parameter_list()}"""

    model = (llm or build_chat_model()).with_structured_output(PaperEvaluation)
    evaluation: PaperEvaluation = model.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    return evaluation.with_computed_score()


def evaluate_paper_with_retry(
    paper: Paper, llm=None, max_attempts: int = MAX_EVALUATION_ATTEMPTS
) -> tuple[PaperEvaluation | None, EvaluationFailure | None]:
    """`evaluate_paper`, isolated and bounded: up to `max_attempts` tries at
    one paper, never more. One paper failing (a transient provider error,
    a rate limit, a structured-output response the model got wrong) must
    not take down a research run that's evaluating nine other papers fine.

    Deliberately catches `Exception` broadly rather than a curated list of
    provider exception classes: the failures worth retrying here span
    pydantic's `ValidationError` (malformed structured output) and
    provider-specific network/rate-limit exceptions from whichever LLM
    backend is configured, and the retry count is capped regardless of
    *what* failed — a broad catch inside a bounded loop is safe in a way a
    broad catch around unbounded work is not.

    No backoff between attempts: three immediate retries are aimed at
    transient/flaky failures, not at riding out a sustained rate limit —
    that would need real backoff and jitter, which is out of scope until
    evidence (see item 9 / ADR follow-up) shows immediate retries aren't
    enough in practice.

    Returns `(evaluation, None)` on success or `(None, failure)` once every
    attempt is exhausted — never raises, so a caller can loop over many
    papers without a per-paper try/except of its own.
    """

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return evaluate_paper(paper, llm=llm), None
        except Exception as exc:  # noqa: BLE001 — see docstring
            last_error = exc
            logger.warning(
                "Evaluation attempt %d/%d failed for paper %s (%s): %s",
                attempt,
                max_attempts,
                paper.id,
                type(exc).__name__,
                exc,
            )

    assert last_error is not None  # the loop always runs at least once
    failure = EvaluationFailure(
        paper_id=paper.id,
        paper_title=paper.title,
        error_type=type(last_error).__name__,
        error_message=str(last_error),
        attempts=max_attempts,
    )
    return None, failure
