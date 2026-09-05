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

import hashlib
import logging
import random
import time
from collections.abc import Callable

import anthropic
import openai
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from agiresearch.domain.schemas import EvaluationFailure, Paper, PaperEvaluation
from agiresearch.domain.scoring import AGI_PARAMETERS
from agiresearch.llm.client import build_chat_model

logger = logging.getLogger(__name__)

MAX_EVALUATION_ATTEMPTS = 3

# Exponential backoff with jitter between retry attempts: attempt 1 -> 2
# waits ~0.5s, attempt 2 -> 3 waits ~1.0s, each plus up to 0.25s of random
# jitter. Small on purpose — three attempts total, aimed at riding out a
# brief provider hiccup, not a sustained outage.
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_JITTER_SECONDS = 0.25

# Retried: a malformed structured-output response (a fluke of the model's
# formatting, likely to succeed on a second try) and known-transient
# provider errors (connection drop, timeout, rate limit, a 5xx on the
# provider's side). Explicitly NOT retried: everything else — an
# AuthenticationError, a BadRequestError, a plain TypeError/AttributeError
# from a bug in this code. None of those are fixed by trying the exact
# same call again, so retrying them would only burn attempts (and, for a
# real provider, money) for no chance of a different outcome.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ValidationError,
    TimeoutError,
    ConnectionError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

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

# Logged alongside every real-model eval result (`evals/result_logging.py`)
# so a later score change can be attributed to a prompt edit rather than
# left as an unexplained number drift — see domain/scoring.py:RUBRIC_VERSION
# for the same reasoning applied to the parameter rubric itself.
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _is_retryable(exc: Exception) -> bool:
    return isinstance(exc, _RETRYABLE_EXCEPTIONS)


def _backoff_seconds(attempt: int, random_fn: Callable[[], float]) -> float:
    return _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random_fn() * _BACKOFF_JITTER_SECONDS


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
    paper: Paper,
    llm=None,
    max_attempts: int = MAX_EVALUATION_ATTEMPTS,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> tuple[PaperEvaluation | None, EvaluationFailure | None]:
    """`evaluate_paper`, isolated and bounded: up to `max_attempts` tries at
    one paper, never more. One paper failing (a transient provider error,
    a rate limit, a structured-output response the model got wrong) must
    not take down a research run that's evaluating nine other papers fine.

    Only retries `_is_retryable` failures (malformed structured output,
    connection/timeout/rate-limit/5xx errors) — see that tuple's comment
    for why. Anything else (an auth error, a bad request, a plain bug in
    this code) fails on the first attempt: retrying an exact repeat of a
    call that can't succeed just burns attempts — and, against a real
    provider, money — for no chance of a different outcome. `attempts` on
    the resulting `EvaluationFailure` reflects how many tries actually
    happened, so a fail-fast case is visibly `attempts=1`, not padded out
    to look like the retries ran.

    Retries wait with exponential backoff plus jitter (`_backoff_seconds`)
    rather than immediately — enough for a brief provider hiccup or a
    rate-limit window to pass, still small since there are only ever two
    waits (before attempt 2, before attempt 3) no matter what.
    `sleep_fn`/`random_fn` default to the real `time.sleep`/`random.random`;
    tests inject no-op/deterministic stand-ins so retry tests run in
    milliseconds, not seconds.

    Returns `(evaluation, None)` on success or `(None, failure)` once
    retries are exhausted or a non-retryable error is hit — never raises,
    so a caller can loop over many papers without a per-paper try/except.
    """

    last_error: Exception | None = None
    attempts_made = 0

    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        try:
            return evaluate_paper(paper, llm=llm), None
        except Exception as exc:  # noqa: BLE001 — classified by _is_retryable below
            last_error = exc
            retryable = _is_retryable(exc)
            logger.warning(
                "Evaluation attempt %d/%d failed for paper %s (%s, retryable=%s): %s",
                attempt,
                max_attempts,
                paper.id,
                type(exc).__name__,
                retryable,
                exc,
            )
            if not retryable:
                break
            if attempt < max_attempts:
                sleep_fn(_backoff_seconds(attempt, random_fn))

    assert last_error is not None  # the loop always runs at least once
    failure = EvaluationFailure(
        paper_id=paper.id,
        paper_title=paper.title,
        error_type=type(last_error).__name__,
        error_message=str(last_error),
        attempts=attempts_made,
    )
    return None, failure
