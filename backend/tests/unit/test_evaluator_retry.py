"""Unit tests for evaluate_paper_with_retry — bounded, isolated retries.

Uses a small stub standing in for `build_chat_model()`'s return value
(anything with a `.with_structured_output(schema).invoke(messages)` shape)
so failures can be injected deterministically without touching a real
provider or the FakeChatModel heuristic.
"""

from __future__ import annotations

from datetime import datetime

import pytest

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


def _valid_evaluation() -> PaperEvaluation:
    scores = ParameterScores(
        **{name: ParameterScore(score=5.0, reasoning="ok") for name in AGI_PARAMETERS}
    )
    return PaperEvaluation(parameter_scores=scores, overall_assessment="fine")


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


def test_first_attempt_fails_second_succeeds():
    # A malformed-structured-output failure (what a real ValidationError from
    # a bad LLM response would raise) is just one instance of "some
    # exception" as far as the retry loop is concerned — see its docstring
    # on why the catch is broad rather than a curated exception list.
    llm = _StubLLM([ValueError("malformed structured output"), _valid_evaluation()])

    evaluation, failure = evaluate_paper_with_retry(PAPER, llm=llm)

    assert failure is None
    assert evaluation is not None
    assert evaluation.agi_score is not None  # with_computed_score() was applied
    assert llm.call_count == 2


def test_permanent_failure_after_max_attempts():
    error = RuntimeError("simulated provider outage")
    llm = _StubLLM([error, error, error])

    evaluation, failure = evaluate_paper_with_retry(PAPER, llm=llm)

    assert evaluation is None
    assert failure is not None
    assert failure.paper_id == PAPER.id
    assert failure.paper_title == PAPER.title
    assert failure.error_type == "RuntimeError"
    assert failure.error_message == "simulated provider outage"
    assert failure.attempts == MAX_EVALUATION_ATTEMPTS
    assert llm.call_count == MAX_EVALUATION_ATTEMPTS


def test_retry_loop_is_bounded_not_unbounded():
    # Four failures queued, but only 3 attempts allowed — if the loop were
    # unbounded this would raise IndexError from popping an empty list.
    error = RuntimeError("still failing")
    llm = _StubLLM([error, error, error, error])

    evaluation, failure = evaluate_paper_with_retry(PAPER, llm=llm, max_attempts=3)

    assert evaluation is None
    assert failure.attempts == 3
    assert llm.call_count == 3


def test_success_on_first_attempt_does_not_retry():
    llm = _StubLLM([_valid_evaluation()])

    evaluation, failure = evaluate_paper_with_retry(PAPER, llm=llm)

    assert failure is None
    assert evaluation is not None
    assert llm.call_count == 1


@pytest.mark.parametrize("max_attempts", [1, 2, 5])
def test_max_attempts_is_configurable(max_attempts):
    error = RuntimeError("nope")
    llm = _StubLLM([error] * max_attempts)

    _, failure = evaluate_paper_with_retry(PAPER, llm=llm, max_attempts=max_attempts)

    assert failure.attempts == max_attempts
    assert llm.call_count == max_attempts
