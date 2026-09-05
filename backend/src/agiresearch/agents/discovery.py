"""Discovery "agent": a direct call into the deterministic arXiv tool.

Not an LLM agent. A ReAct tool-calling agent whose only real job is
invoking one tool once, with arguments the execution plan has already
fully determined, adds an LLM decision point (call the tool? with what
args? how many times?) where none is needed. Calling `tools.arxiv_search`
directly here removes that round-trip and its failure modes entirely. See
docs/adr/0002-direct-tool-call-instead-of-react-agent.md.
"""

from __future__ import annotations

from agiresearch.domain.schemas import DiscoveryResult, ExecutionPlan
from agiresearch.tools.arxiv_search import discover_and_process_papers


def discover_papers(plan: ExecutionPlan) -> DiscoveryResult:
    return discover_and_process_papers(
        keywords=plan.search_keywords,
        categories=plan.categories,
        date_from=plan.date_from,
        date_to=plan.date_to,
        max_papers=plan.max_papers,
    )
