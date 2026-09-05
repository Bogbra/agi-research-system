"""Weighted AGI-potential scoring: pure functions, no LLM or LangGraph import.

Kept deterministic and framework-agnostic on purpose: this is the one part
of the pipeline whose correctness doesn't depend on a model's mood, so it
should be testable with plain `pytest`, no fake LLM required, no network.

The LLM's only job (see `agents/evaluator.py`) is producing a 1-10 score per
parameter, with reasoning, as a structured output. Combining those into a
final 0-100 score and a classification band is arithmetic, not judgment, and
belongs here.
"""

from __future__ import annotations

from typing import NamedTuple


class ParameterWeight(NamedTuple):
    weight: float
    description: str


# Bumped whenever AGI_PARAMETERS' set of parameters or their weights change
# structurally. Logged alongside every real-model eval result
# (`evals/result_logging.py`) so a later score change can be attributed to
# a rubric change, a prompt change (`agents/evaluator.py:PROMPT_SHA256`),
# or a model change — not left as an unexplained number drift.
RUBRIC_VERSION = "1.0.0"


# Single source of truth for parameter names and weights — domain/schemas.py
# imports AGI_PARAMETERS.keys() to validate that a judged evaluation scored
# every parameter, rather than duplicating this list a second time.
AGI_PARAMETERS: dict[str, ParameterWeight] = {
    "novel_problem_solving": ParameterWeight(0.15, "Solving new, unseen problems"),
    "few_shot_learning": ParameterWeight(0.15, "Learning from minimal examples"),
    "task_transfer": ParameterWeight(0.15, "Applying skills across domains"),
    "abstract_reasoning": ParameterWeight(0.12, "Logical thinking & pattern recognition"),
    "contextual_adaptation": ParameterWeight(0.10, "Adapting behavior to context"),
    "multi_rule_integration": ParameterWeight(0.10, "Following multiple complex rules"),
    "generalization_efficiency": ParameterWeight(0.08, "Generalizing from small data"),
    "meta_learning": ParameterWeight(0.08, "Learning how to learn"),
    "world_modeling": ParameterWeight(0.04, "Modeling complex environments"),
    "autonomous_goal_setting": ParameterWeight(0.03, "Setting & pursuing own objectives"),
}

assert abs(sum(p.weight for p in AGI_PARAMETERS.values()) - 1.0) < 1e-9, (
    "AGI_PARAMETERS weights must sum to 1.0"
)

HIGH_POTENTIAL_THRESHOLD = 70.0
MEDIUM_POTENTIAL_THRESHOLD = 40.0


class ScoreBreakdown(NamedTuple):
    final_score: float
    classification: str
    contributions: dict[str, float]


def calculate_agi_score(parameter_scores: dict[str, float]) -> ScoreBreakdown:
    """Weighted AGI score (0-100) from per-parameter scores (each 1-10).

    `parameter_scores` must cover every key in `AGI_PARAMETERS` — this is
    enforced upstream by `domain.schemas.PaperEvaluation`'s validator, not
    re-checked here, so a caller bypassing that schema gets a `KeyError`
    rather than a silently partial score.
    """

    contributions = {
        name: round(parameter_scores[name] * weight.weight, 2)
        for name, weight in AGI_PARAMETERS.items()
    }
    final = round(sum(contributions.values()) * 10, 1)

    if final >= HIGH_POTENTIAL_THRESHOLD:
        classification = "High AGI Potential"
    elif final >= MEDIUM_POTENTIAL_THRESHOLD:
        classification = "Medium AGI Potential"
    else:
        classification = "Low AGI Potential"

    return ScoreBreakdown(
        final_score=final, classification=classification, contributions=contributions
    )
