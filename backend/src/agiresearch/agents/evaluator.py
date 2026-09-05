"""Evaluator agent: judges one paper's AGI-advancement potential.

Replaces a three-tier manual JSON recovery (markdown-fence strip, then
trailing-comma repair, then regex-scraping individual scores out of broken
JSON) with a single `with_structured_output(PaperEvaluation)` call. See
docs/adr/0001-structured-agent-outputs.md.

The weighted 0-100 score is computed afterward by
`PaperEvaluation.with_computed_score()` (pure arithmetic, `domain/scoring.py`)
rather than asked of the model — a judge asked to also do the weighted-sum
arithmetic is a judge that can get the arithmetic wrong on top of whatever
else it gets wrong. See docs/adr/0006-agi-judge-evaluation.md.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agiresearch.domain.schemas import Paper, PaperEvaluation
from agiresearch.domain.scoring import AGI_PARAMETERS
from agiresearch.llm.client import build_chat_model

SYSTEM_PROMPT = """You are an expert AGI evaluator. Analyze research papers for \
their contribution to Artificial General Intelligence.

Rate each of the 10 AGI parameters on a 1-10 scale:
- 1-3: no/minimal AGI relevance
- 4-6: some AGI potential but limited
- 7-8: strong AGI contribution
- 9-10: exceptional AGI breakthrough

Be conservative — reserve 7+ for truly exceptional work. Focus on what the
paper actually demonstrates, not just claims. Distinguish narrow-AI
improvements from genuine AGI progress."""


def _parameter_list() -> str:
    return "\n".join(
        f"- {name} ({weight.weight:.0%}): {weight.description}"
        for name, weight in AGI_PARAMETERS.items()
    )


def evaluate_paper(paper: Paper, llm=None) -> PaperEvaluation:
    """Judge `paper` against the 10-parameter AGI rubric and compute its
    weighted score.

    `llm` defaults to `build_chat_model()` — pass one explicitly to reuse a
    single instance across many calls instead of hitting the (cached)
    factory each time.
    """

    authors = ", ".join(paper.metadata.authors[:5])
    user_prompt = f"""TITLE:
{paper.title}

AUTHORS:
{authors}

ABSTRACT:
{paper.metadata.abstract}

AGI PARAMETERS TO SCORE:
{_parameter_list()}"""

    model = (llm or build_chat_model()).with_structured_output(PaperEvaluation)
    evaluation: PaperEvaluation = model.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    return evaluation.with_computed_score()
