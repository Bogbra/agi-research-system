"""Research run submission and retrieval."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from agiresearch.api.auth import Principal, get_current_principal
from agiresearch.api.db import ResearchRunRecord, get_db
from agiresearch.api.schemas_api import (
    EvaluatedPaperView,
    RunAcceptedResponse,
    RunDetail,
    RunSummary,
    SubmitResearchRequest,
)
from agiresearch.domain.schemas import ResearchPhase
from agiresearch.orchestrator import run_research

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


def _execute_run(request_id: str, objective: str) -> None:
    """Runs in a worker thread via BackgroundTasks; owns its own DB session."""
    from agiresearch.api.db import SessionLocal

    db = SessionLocal()
    try:
        final_state = run_research(objective)
        # run_research assigns its own request_id (a fresh uuid4) inside the
        # graph; the DB row is keyed by the id generated at submission time,
        # so overwrite it before persisting to keep the two in sync.
        final_state = final_state.model_copy(update={"request_id": request_id})

        record = db.get(ResearchRunRecord, request_id)
        if record is None:
            return

        scores = [
            p.evaluation.agi_score
            for p in final_state.evaluated_papers
            if p.evaluation.agi_score is not None
        ]
        record.status = final_state.current_phase.value
        record.paper_count = len(final_state.evaluated_papers)
        record.average_agi_score = round(sum(scores) / len(scores), 1) if scores else None
        record.final_report = final_state.final_report
        record.state_json = final_state.model_dump(mode="json")
        db.add(record)
        db.commit()
    except Exception as exc:  # noqa: BLE001 — surfaced to the run record, not swallowed
        logger.exception("Research run failed for %s", request_id)
        record = db.get(ResearchRunRecord, request_id)
        if record is not None:
            record.status = "failed"
            record.error = str(exc)
            db.add(record)
            db.commit()
    finally:
        db.close()


@router.post("", response_model=RunAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_research(
    payload: SubmitResearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _principal: Principal = Depends(get_current_principal),
) -> RunAcceptedResponse:
    request_id = str(uuid.uuid4())
    record = ResearchRunRecord(
        request_id=request_id,
        research_objective=payload.objective,
        status=ResearchPhase.INITIALIZATION.value,
        state_json={},
    )
    db.add(record)
    db.commit()

    background_tasks.add_task(_execute_run, request_id, payload.objective)
    return RunAcceptedResponse(request_id=request_id, status=record.status)


@router.get("", response_model=list[RunSummary])
def list_research_runs(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _principal: Principal = Depends(get_current_principal),
) -> list[RunSummary]:
    limit = max(1, min(limit, 200))
    records = (
        db.execute(
            select(ResearchRunRecord)
            .order_by(ResearchRunRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return [
        RunSummary(
            request_id=r.request_id,
            research_objective=r.research_objective,
            status=r.status,
            paper_count=r.paper_count,
            average_agi_score=r.average_agi_score,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in records
    ]


@router.get("/{request_id}", response_model=RunDetail)
def get_research_run(
    request_id: str,
    db: Session = Depends(get_db),
    _principal: Principal = Depends(get_current_principal),
) -> RunDetail:
    record = db.get(ResearchRunRecord, request_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Research run {request_id} not found.")

    state = record.state_json or {}
    evaluated_papers = [
        EvaluatedPaperView(
            title=item["paper"]["title"],
            link=item["paper"]["link"],
            authors=item["paper"]["metadata"]["authors"],
            agi_score=item["evaluation"]["agi_score"],
            classification=item["evaluation"]["classification"],
            overall_assessment=item["evaluation"]["overall_assessment"],
            key_innovations=item["evaluation"]["key_innovations"],
        )
        for item in state.get("evaluated_papers", [])
    ]

    return RunDetail(
        request_id=record.request_id,
        research_objective=record.research_objective,
        status=record.status,
        paper_count=record.paper_count,
        average_agi_score=record.average_agi_score,
        final_report=record.final_report,
        evaluated_papers=evaluated_papers,
        errors=state.get("errors", []),
        error=record.error,
        execution_plan=state.get("execution_plan"),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
