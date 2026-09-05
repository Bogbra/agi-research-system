"""Markdown report rendering — pure string formatting over `ResearchState`.

No LLM involved. Kept separate from `graph/build.py` so it's testable as a
plain function of already-computed data, and separate from `domain/` since
it's presentation, not business logic.
"""

from __future__ import annotations

from agiresearch.domain.schemas import ResearchState
from agiresearch.domain.scoring import HIGH_POTENTIAL_THRESHOLD, MEDIUM_POTENTIAL_THRESHOLD


def render_final_report(state: ResearchState, top_n: int = 5) -> str:
    evaluated = state.evaluated_papers

    if not evaluated:
        return (
            f"# AGI Research Report\n\n"
            f"**Objective:** {state.research_objective}\n\n"
            f"No papers were successfully evaluated.\n"
        )

    ranked = sorted(evaluated, key=lambda e: e.evaluation.agi_score or 0.0, reverse=True)
    scores = [e.evaluation.agi_score or 0.0 for e in evaluated]
    average = round(sum(scores) / len(scores), 1)
    high = sum(1 for s in scores if s >= HIGH_POTENTIAL_THRESHOLD)
    medium = sum(1 for s in scores if MEDIUM_POTENTIAL_THRESHOLD <= s < HIGH_POTENTIAL_THRESHOLD)
    low = len(scores) - high - medium

    lines = [
        "# AGI Research Report",
        "",
        f"**Objective:** {state.research_objective}",
        "",
        "## Discovery Overview",
        f"- Papers discovered: {len(state.discovered_papers)}",
        f"- Papers evaluated: {len(evaluated)}",
        f"- Average AGI score: {average}/100",
        "",
        "## AGI Potential Distribution",
        f"- High (>= {HIGH_POTENTIAL_THRESHOLD:.0f}): {high}",
        f"- Medium ({MEDIUM_POTENTIAL_THRESHOLD:.0f}-{HIGH_POTENTIAL_THRESHOLD:.0f}): {medium}",
        f"- Low (< {MEDIUM_POTENTIAL_THRESHOLD:.0f}): {low}",
        "",
        "## Top Papers",
        "",
    ]

    for i, item in enumerate(ranked[:top_n], 1):
        paper, evaluation = item.paper, item.evaluation
        authors = ", ".join(paper.metadata.authors[:3])
        innovations = ", ".join(evaluation.key_innovations[:3]) or "n/a"
        lines += [
            f"### {i}. {paper.title}",
            f"**Authors:** {authors}",
            f"**AGI score:** {evaluation.agi_score}/100 ({evaluation.classification})",
            f"**Key innovations:** {innovations}",
            f"**Assessment:** {evaluation.overall_assessment}",
            f"**Link:** {paper.link}",
            "",
        ]

    return "\n".join(lines)
