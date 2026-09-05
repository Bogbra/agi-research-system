"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agiresearch.api.db import init_db
from agiresearch.api.rate_limit import RateLimitMiddleware
from agiresearch.api.routers import health, research
from agiresearch.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AGI Research Intelligence API",
    description=(
        "Agentic research pipeline (LLM planning + evaluation, deterministic discovery "
        "and scoring): submission, discovery, evaluation, and retrieval."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60, burst=20)

app.include_router(health.router)
app.include_router(research.router)
