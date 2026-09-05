"""Unit tests for the offline FakeChatModel — proves the harness runs with
no API key, not that the heuristics are a meaningful judgment (see
llm/fake.py's own docstring).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agiresearch.domain.schemas import ExecutionPlan, PaperEvaluation
from agiresearch.domain.scoring import AGI_PARAMETERS
from agiresearch.llm.fake import FakeChatModel


def test_fake_model_produces_valid_execution_plan():
    llm = FakeChatModel().with_structured_output(ExecutionPlan)
    content = (
        "OBJECTIVE:\nfind agi papers about meta learning\n\nTODAY:\n2026-01-31\n\nLOOKBACK_DAYS:\n7"
    )
    plan = llm.invoke([HumanMessage(content=content)])
    assert isinstance(plan, ExecutionPlan)
    assert plan.date_to.isoformat() == "2026-01-31"
    assert plan.date_from.isoformat() == "2026-01-24"
    assert "meta" in plan.search_keywords


def test_fake_model_falls_back_to_default_keywords_on_empty_objective():
    llm = FakeChatModel().with_structured_output(ExecutionPlan)
    plan = llm.invoke([HumanMessage(content="OBJECTIVE:\n\n")])
    assert plan.search_keywords  # never empty — schema requires min_length=1


def test_fake_model_produces_valid_paper_evaluation_with_all_parameters():
    llm = FakeChatModel().with_structured_output(PaperEvaluation)
    evaluation = llm.invoke(
        [
            HumanMessage(
                content="ABSTRACT:\nA study of few-shot learning and meta-learning in transformers."
            )
        ]
    )
    assert isinstance(evaluation, PaperEvaluation)
    assert set(evaluation.parameter_scores.as_score_dict()) == set(AGI_PARAMETERS)
    assert (
        evaluation.parameter_scores.few_shot_learning.score
        > evaluation.parameter_scores.world_modeling.score
    )


def test_fake_model_rejects_unknown_schema():
    llm = FakeChatModel().with_structured_output(dict)
    try:
        llm.invoke([HumanMessage(content="anything")])
        raised = False
    except TypeError:
        raised = True
    assert raised
