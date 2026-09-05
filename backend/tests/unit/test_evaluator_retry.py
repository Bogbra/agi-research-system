"""Unit tests for evaluate_paper_with_retry — bounded, classified,
backed-off retries.

Uses a small stub standing in for `build_chat_model()`'s return value
(anything with a `.with_structured_output(schema).invoke(messages)` shape)
so failures can be injected deterministically without touching a real
provider or the FakeChatModel heuristic. `sleep_fn`/`random_fn` are always
overridden so these tests run in milliseconds even though the real
function backs off between attempts.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import openai
import pytest
from pydantic import ValidationError

from agiresearch.agents.evaluator import MAX_EVALUATION_ATTEMPTS, evaluate_paper_with_retry
from agiresearch.domain.schemas import (
    Paper,
    PaperEvaluation,
    PaperMetadata,
    ParameterScore,
    ParameterScores,
)
from agiresearch.domain.scoring import AGI_PARAMETERS

PAPER = Paper(
    id="1234.5678",
    title="A Test Paper",
    link="https://arxiv.org/abs/1234.5678",
    metadata=PaperMetadata(
        authors=["A. Researcher"],
        abstract="An abstract long enough to pass validation elsewhere.",
        published_date=datetime(2026, 1, 1),
        categories=["cs.AI"],
    ),
)

# Never actually slept on — every call in this file passes no-op stand-ins.
_NO_SLEEP = lambda seconds: None  # noqa: E731
_NO_JITTER = lambda: 0.0  # noqa: E731


def _valid_evaluation() -> PaperEvaluation:
    scores = ParameterScores(
        **{name: ParameterScore(score=5.0, reasoning="ok") for name in AGI_PARAMETERS}
    )
    return PaperEvaluation(parameter_scores=scores, overall_assessment="fine")


def _real_validation_error() -> ValidationError:
    """A genuine `pydantic.ValidationError`, not a hand-built stand-in —
    triggered by actually violating `ParameterScore`'s `score <= 10`
    constraint, so this is exactly the shape a malformed structured-output
    response from a real provider would raise."""

    try:
        ParameterScore(score=999, reasoning="out of range")
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ParameterScore(score=999, ...) to raise ValidationError")


class _StubRunnable:
    def __init__(self, responses: list[BaseException | PaperEvaluation]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _StubLLM:
    def __init__(self, responses: list[BaseException | PaperEvaluation]) -> None:
        self._runnable = _StubRunnable(responses)

    def with_structured_output(self, schema):
        return self._runnable

    @property
    def call_count(self) -> int:
        return self._runnable.call_count


def _retry(llm, **kwargs):
    return evaluate_paper_with_retry(
        PAPER, llm=llm, sleep_fn=_NO_SLEEP, random_fn=_NO_JITTER, **kwargs
    )


def test_first_attempt_fails_second_succeeds():
    # A malformed-structured-output failure — the real ValidationError a
    # provider's structured-output response would raise if the model
    # returned an out-of-range score — is retryable, unlike a plain bug.
    llm = _StubLLM([_real_validation_error(), _valid_evaluation()])

    evaluation, failure = _retry(llm)

    assert failure is None
    assert evaluation is not None
    assert evaluation.agi_score is not None  # with_computed_score() was applied
    assert llm.call_count == 2


def test_permanent_retryable_failure_exhausts_all_attempts():
    error = openai.APITimeoutError(request=None)
    llm = _StubLLM([error, error, error])

    evaluation, failure = _retry(llm)

    assert evaluation is None
    assert failure is not None
    assert failure.paper_id == PAPER.id
    assert failure.paper_title == PAPER.title
    assert failure.error_type == "APITimeoutError"
    assert failure.attempts == MAX_EVALUATION_ATTEMPTS
    assert llm.call_count == MAX_EVALUATION_ATTEMPTS


def test_retry_loop_is_bounded_not_unbounded():
    # Four failures queued, but only 3 attempts allowed — if the loop were
    # unbounded this would raise IndexError from popping an empty list.
    error = ConnectionError("still failing")
    llm = _StubLLM([error, error, error, error])

    evaluation, failure = _retry(llm, max_attempts=3)

    assert evaluation is None
    assert failure.attempts == 3
    assert llm.call_count == 3


def test_success_on_first_attempt_does_not_retry():
    llm = _StubLLM([_valid_evaluation()])

    evaluation, failure = _retry(llm)

    assert failure is None
    assert evaluation is not None
    assert llm.call_count == 1


@pytest.mark.parametrize("max_attempts", [1, 2, 5])
def test_max_attempts_is_configurable_for_retryable_errors(max_attempts):
    error = ConnectionError("nope")
    llm = _StubLLM([error] * max_attempts)

    _, failure = _retry(llm, max_attempts=max_attempts)

    assert failure.attempts == max_attempts
    assert llm.call_count == max_attempts


def test_non_retryable_error_fails_fast_without_burning_attempts():
    # A plain TypeError is an obvious bug, not a transient failure — no
    # number of retries would make the same call succeed, so it must not
    # be retried 3 times just because 3 is the configured max.
    queued = [TypeError("unexpected keyword argument"), _valid_evaluation(), _valid_evaluation()]
    llm = _StubLLM(queued)

    evaluation, failure = _retry(llm)

    assert evaluation is None
    assert failure is not None
    assert failure.error_type == "TypeError"
    assert failure.attempts == 1
    assert llm.call_count == 1  # never touched the queued successes


def test_authentication_error_is_not_retried():
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/x"))
    error = openai.AuthenticationError(message="invalid api key", response=response, body=None)
    llm = _StubLLM([error, _valid_evaluation()])

    evaluation, failure = _retry(llm)

    assert evaluation is None
    assert failure.attempts == 1
    assert llm.call_count == 1


def test_backoff_uses_injected_sleep_and_grows_exponentially():
    sleeps: list[float] = []
    error = ConnectionError("flaky")
    llm = _StubLLM([error, error, _valid_evaluation()])

    evaluation, failure = evaluate_paper_with_retry(
        PAPER,
        llm=llm,
        sleep_fn=lambda seconds: sleeps.append(seconds),
        random_fn=lambda: 0.0,  # zero jitter for a deterministic assertion
    )

    assert failure is None
    assert evaluation is not None
    # Two waits (before attempt 2, before attempt 3), strictly increasing —
    # exponential backoff, not a fixed delay.
    assert len(sleeps) == 2
    assert sleeps[0] == pytest.approx(0.5)
    assert sleeps[1] == pytest.approx(1.0)
    assert sleeps[1] > sleeps[0]


def test_no_sleep_after_the_final_exhausted_attempt():
    sleeps: list[float] = []
    error = ConnectionError("still flaky")
    llm = _StubLLM([error, error, error])

    evaluate_paper_with_retry(
        PAPER,
        llm=llm,
        sleep_fn=lambda seconds: sleeps.append(seconds),
        random_fn=lambda: 0.0,
    )

    # Only 2 waits for 3 attempts — no point sleeping after giving up.
    assert len(sleeps) == 2
