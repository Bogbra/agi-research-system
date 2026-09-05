"""Persistence: one row per research run.

SQLite by default (zero-setup local dev); point `DATABASE_URL` at Postgres
for anything shared. No audit-log table here, unlike a domain with a
compliance/approval step to trace — this pipeline has no human-in-the-loop
action to audit, so a second table for it would be a fabricated concept,
not a proportional one.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from agiresearch.config import settings


class Base(DeclarativeBase):
    pass


class ResearchRunRecord(Base):
    __tablename__ = "research_runs"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    research_objective: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, index=True)  # ResearchPhase value, or "failed"
    paper_count: Mapped[int] = mapped_column(Integer, default=0)
    average_agi_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    state_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
