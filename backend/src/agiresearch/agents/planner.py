"""Planner agent: turns a research objective into a typed execution plan.

Replaces markdown-fence stripping plus `json.loads` with a hand-written
default-plan fallback on parse failure with a single
`with_structured_output(ExecutionPlan)` call — a malformed response is a
validation error the workflow can catch, not a silently substituted
default plan. See docs/adr/0001-structured-agent-outputs.md.
"""

from __future__ import annotations

from datetime import date, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from agiresearch.config import settings
from agiresearch.domain.schemas import ExecutionPlan
from agiresearch.llm.client import build_chat_model

SYSTEM_PROMPT = """You are a Research Planning Specialist for an AGI research \
intelligence system.

Given a research objective, produce a structured execution plan: search
keywords, arXiv categories, a date range, and focus areas.

Derive the date range strictly from the objective's own wording (e.g. "last
one week" -> 7 days back from today, "last month" -> 30 days back). If no
time period is mentioned, use the default lookback given below.

Extract 5-8 relevant search keywords and 2-4 specific focus areas."""


def plan_research(objective: str, llm=None) -> ExecutionPlan:
    """Produce an `ExecutionPlan` for `objective`.

    `llm` defaults to `build_chat_model()` — pass one explicitly to reuse a
    single instance across many calls instead of hitting the (cached)
    factory each time.
    """

    today = date.today()
    user_prompt = (
        f"OBJECTIVE:\n{objective}\n\n"
        f"TODAY:\n{today.isoformat()}\n\n"
        f"LOOKBACK_DAYS:\n{settings.default_lookback_days}\n\n"
        f"Reference dates — 7 days ago: {(today - timedelta(days=7)).isoformat()}, "
        f"14 days ago: {(today - timedelta(days=14)).isoformat()}, "
        f"30 days ago: {(today - timedelta(days=30)).isoformat()}."
    )

    model = (llm or build_chat_model()).with_structured_output(ExecutionPlan)
    return model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
