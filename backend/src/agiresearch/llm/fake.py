"""Deterministic, offline stand-in for a chat model.

Two things this project should never require just to be evaluated: a paid
API key, or a network call. `LLM_PROVIDER=fake` (the default) routes through
this module instead, so the whole pipeline — planning, discovery, scoring —
runs end to end with synthetic-but-deterministic reasoning in place of a
real LLM call. Golden-case regression tests run against this model for the
same reason: no flakiness, no API cost, no network dependency.

Swapping to a real provider is a one-line env change (`LLM_PROVIDER=openai`)
— see agiresearch/llm/client.py. The heuristics below are keyword-matching,
not semantic understanding — they exist to prove the harness runs, not to
produce a meaningful AGI judgment (that's the whole reason
`evals/judge_reliability.py` insists on a real provider before trusting a
score).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from agiresearch.config import settings
from agiresearch.domain.schemas import (
    ExecutionPlan,
    PaperEvaluation,
    ParameterScore,
    ParameterScores,
)
from agiresearch.domain.scoring import AGI_PARAMETERS

_OBJECTIVE_RE = re.compile(r"OBJECTIVE:\s*(.*?)\s*(?:\n\n|\Z)", re.DOTALL)
_TODAY_RE = re.compile(r"TODAY:\s*(\d{4}-\d{2}-\d{2})")
_LOOKBACK_RE = re.compile(r"LOOKBACK_DAYS:\s*(\d+)")
_ABSTRACT_RE = re.compile(r"ABSTRACT:\s*(.*?)\s*\Z", re.DOTALL)
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")

# Loose keyword hints per parameter — a fake judge's approximation of "does
# this abstract even mention the concept", not a scoring rubric.
_PARAMETER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "novel_problem_solving": ("novel", "unseen problem", "new problem"),
    "few_shot_learning": ("few-shot", "few shot", "in-context learning"),
    "task_transfer": ("transfer learning", "cross-domain", "task transfer"),
    "abstract_reasoning": ("abstract reasoning", "logical reasoning", "reasoning"),
    "contextual_adaptation": ("adapt", "context-aware", "contextual"),
    "multi_rule_integration": ("multi-rule", "rule integration", "constraint"),
    "generalization_efficiency": ("generaliz", "sample-efficien", "data-efficien"),
    "meta_learning": ("meta-learning", "meta learning", "learning to learn"),
    "world_modeling": ("world model", "environment model", "simulation"),
    "autonomous_goal_setting": ("autonomous", "self-directed", "goal setting"),
}

_KEYWORD_MATCH_SCORE = 6.0
_NO_MATCH_SCORE = 3.0


def _messages_to_text(messages: Sequence[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages)


def _build_plan(text: str) -> ExecutionPlan:
    objective_match = _OBJECTIVE_RE.search(text)
    objective = objective_match.group(1) if objective_match else text

    today_match = _TODAY_RE.search(text)
    today = date.fromisoformat(today_match.group(1)) if today_match else date.today()

    lookback_match = _LOOKBACK_RE.search(text)
    lookback_days = (
        int(lookback_match.group(1)) if lookback_match else settings.default_lookback_days
    )

    seen: list[str] = []
    for word in _WORD_RE.findall(objective.lower()):
        if word not in seen:
            seen.append(word)
    keywords = seen[:8] or ["agi", "general intelligence"]

    return ExecutionPlan(
        search_keywords=keywords,
        categories=list(settings.arxiv_categories),
        date_from=today - timedelta(days=lookback_days),
        date_to=today,
        max_papers=settings.default_max_papers,
        focus_areas=keywords[:3],
    )


def _build_evaluation(text: str) -> PaperEvaluation:
    abstract_match = _ABSTRACT_RE.search(text)
    abstract = (abstract_match.group(1) if abstract_match else text).lower()

    scores: dict[str, ParameterScore] = {}
    matched = 0
    for name in AGI_PARAMETERS:
        hit = any(kw in abstract for kw in _PARAMETER_KEYWORDS[name])
        matched += hit
        scores[name] = ParameterScore(
            score=_KEYWORD_MATCH_SCORE if hit else _NO_MATCH_SCORE,
            reasoning=(
                "[demo-mode] matched a keyword hint"
                if hit
                else "[demo-mode] no keyword hint matched"
            ),
        )

    return PaperEvaluation(
        parameter_scores=ParameterScores(**scores),
        overall_assessment=(
            f"[demo-mode] keyword heuristic matched {matched}/{len(AGI_PARAMETERS)} parameter "
            "hint group(s) — not a real judgment. Set LLM_PROVIDER=openai for a real evaluation."
        ),
        confidence="Low",
    )


class FakeChatModel(BaseChatModel):
    """A `BaseChatModel` that never leaves the process.

    Only `with_structured_output` is meaningfully implemented — that is the
    only mode the agent workflow uses. Plain `.invoke()` returns a labelled
    placeholder so misuse is obvious rather than silently wrong.
    """

    @property
    def _llm_type(self) -> str:
        return "fake-deterministic"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        message = AIMessage(content="[demo-mode] FakeChatModel does not generate free text.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema: type, **kwargs: Any) -> Runnable:
        def _invoke(messages: Sequence[BaseMessage]):
            text = _messages_to_text(messages)
            if schema is ExecutionPlan:
                return _build_plan(text)
            if schema is PaperEvaluation:
                return _build_evaluation(text)
            raise TypeError(f"FakeChatModel has no deterministic handler for {schema!r}")

        return RunnableLambda(_invoke)
