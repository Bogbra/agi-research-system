"""Typed contracts for research state, plans, and evaluation output.

Every LLM call in this system binds to one of these Pydantic models via
structured output (`llm.with_structured_output(...)`), so a malformed
response is a validation error the workflow can catch — never a value
scraped out of free text with a regex and a markdown-fence strip. See
docs/adr/0001-structured-agent-outputs.md.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from agiresearch.domain.scoring import AGI_PARAMETERS, calculate_agi_score

# --------------------------------------------------------------------------
# Workflow phases
# --------------------------------------------------------------------------


class ResearchPhase(StrEnum):
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    DISCOVERY = "discovery"
    EVALUATION = "evaluation"
    COMPLETION = "completion"


# --------------------------------------------------------------------------
# Planning (Planner agent's structured output)
# --------------------------------------------------------------------------


class ExecutionPlan(BaseModel):
    """The planner's output: what to search for, where, and over what window.

    Flat and typed on purpose, rather than a nested dict of loosely-typed
    fields that has to be parsed back out by hand at each call site.
    """

    search_keywords: list[str] = Field(min_length=1, max_length=8)
    categories: list[str] = Field(default_factory=list)
    date_from: date
    date_to: date
    max_papers: int = Field(default=10, ge=1, le=50)
    focus_areas: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _date_range_is_ordered(self) -> ExecutionPlan:
        if self.date_from > self.date_to:
            raise ValueError(f"date_from ({self.date_from}) is after date_to ({self.date_to})")
        return self


# --------------------------------------------------------------------------
# Discovery (deterministic tool output — no LLM involved)
# --------------------------------------------------------------------------


class PaperMetadata(BaseModel):
    authors: list[str] = Field(default_factory=list)
    abstract: str
    published_date: datetime
    categories: list[str] = Field(default_factory=list)
    source: str = "arxiv"


class Paper(BaseModel):
    id: str
    title: str
    link: str
    metadata: PaperMetadata


class DiscoveryStats(BaseModel):
    initial_count: int
    after_deduplication: int
    final_count: int
    duplicates_removed: int
    invalid_removed: int


class DiscoveryResult(BaseModel):
    papers: list[Paper]
    stats: DiscoveryStats


# --------------------------------------------------------------------------
# Evaluation (Evaluator agent's structured output)
# --------------------------------------------------------------------------


class ParameterScore(BaseModel):
    score: float = Field(ge=1, le=10)
    reasoning: str = Field(max_length=300)


class ParameterScores(BaseModel):
    """One `ParameterScore` per entry in `domain.scoring.AGI_PARAMETERS`,
    as ten explicit named fields rather than a `dict[str, ParameterScore]`.

    Not a style choice: OpenAI's strict structured-output mode rejects an
    open-ended dict schema outright (`'required' ... must include every key
    in properties` — there's no fixed `properties` set for an arbitrary-key
    dict to satisfy that against). Explicit fields are also what makes a
    missing parameter a validation error instead of a silently incomplete
    dict, matching every other model in this module. Field names are
    `AGI_PARAMETERS` keys and must stay in sync with it — a mismatch fails
    fast in the tests, not in a live judge call.
    """

    model_config = {"extra": "forbid"}

    novel_problem_solving: ParameterScore
    few_shot_learning: ParameterScore
    task_transfer: ParameterScore
    abstract_reasoning: ParameterScore
    contextual_adaptation: ParameterScore
    multi_rule_integration: ParameterScore
    generalization_efficiency: ParameterScore
    meta_learning: ParameterScore
    world_modeling: ParameterScore
    autonomous_goal_setting: ParameterScore

    def as_score_dict(self) -> dict[str, float]:
        return {name: getattr(self, name).score for name in AGI_PARAMETERS}


class PaperEvaluation(BaseModel):
    """One paper's AGI-potential judgment. `agi_score`/`classification` are
    filled in by `calculate_agi_score` after the LLM call returns — they are
    arithmetic on `parameter_scores`, not something the model should report
    itself (a judge asked to also do the weighted-sum arithmetic is a judge
    that can get the arithmetic wrong; see docs/adr/0006).
    """

    parameter_scores: ParameterScores
    overall_assessment: str = Field(max_length=500)
    key_innovations: list[str] = Field(default_factory=list, max_length=5)
    limitations: list[str] = Field(default_factory=list, max_length=5)
    confidence: str = "Medium"

    agi_score: float | None = None
    classification: str | None = None

    def with_computed_score(self) -> PaperEvaluation:
        breakdown = calculate_agi_score(self.parameter_scores.as_score_dict())
        return self.model_copy(
            update={"agi_score": breakdown.final_score, "classification": breakdown.classification}
        )


class EvaluatedPaper(BaseModel):
    """A `Paper` plus its `PaperEvaluation` — what a research run actually
    ranks and reports on."""

    paper: Paper
    evaluation: PaperEvaluation


# --------------------------------------------------------------------------
# Run-level summary (used by the orchestrator, evals, and the API layer)
# --------------------------------------------------------------------------


class RunStatistics(BaseModel):
    total_discovered: int
    total_evaluated: int
    average_agi_score: float
    high_potential_count: int
    medium_potential_count: int
    low_potential_count: int


# --------------------------------------------------------------------------
# LangGraph state
# --------------------------------------------------------------------------


class ResearchState(BaseModel):
    """Full state threaded through the LangGraph research pipeline for one run.

    A plain Pydantic model, not a TypedDict, matching this project's usual
    convention for LangGraph state — see `docs/adr/0001`. No `Annotated[...,
    reducer]` fields are needed here (unlike a graph with a parallel
    fan-out/fan-in): every phase runs strictly one after another, so plain
    field replacement on each node's return is all LangGraph needs to merge
    correctly.
    """

    model_config = {"arbitrary_types_allowed": True}

    request_id: str
    research_objective: str
    current_phase: ResearchPhase = ResearchPhase.INITIALIZATION

    execution_plan: ExecutionPlan | None = None
    discovered_papers: list[Paper] = Field(default_factory=list)
    evaluated_papers: list[EvaluatedPaper] = Field(default_factory=list)

    final_report: str | None = None
    errors: list[str] = Field(default_factory=list)
