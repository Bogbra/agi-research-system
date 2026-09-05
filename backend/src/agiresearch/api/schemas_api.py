"""API request/response contracts — kept separate from the internal domain
schemas so the pipeline's internal state shape can change without breaking
API consumers (the Next.js dashboard, or anyone else)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SubmitResearchRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=500)


class RunAcceptedResponse(BaseModel):
    request_id: str
    status: str


class RunSummary(BaseModel):
    request_id: str
    research_objective: str
    status: str
    paper_count: int
    average_agi_score: float | None
    created_at: datetime
    updated_at: datetime


class EvaluatedPaperView(BaseModel):
    title: str
    link: str
    authors: list[str]
    agi_score: float | None
    classification: str | None
    overall_assessment: str
    key_innovations: list[str]


class RunDetail(BaseModel):
    request_id: str
    research_objective: str
    status: str
    paper_count: int
    average_agi_score: float | None
    final_report: str | None
    evaluated_papers: list[EvaluatedPaperView]
    errors: list[str]
    error: str | None = None
    execution_plan: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
