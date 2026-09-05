from __future__ import annotations

from fastapi import APIRouter

from agiresearch.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "arxiv_provider": settings.arxiv_provider,
    }
