"""Judge reliability: is the AGI-evaluation judge itself trustworthy?

Two independent questions, neither answered by the golden-case eval
(`evals/run.py`) alone:

  Calibration — does the judge score substance over vocabulary? A judge
  asked to "focus on what the paper actually demonstrates, not just
  claims" can still be fooled by a hyped-up abstract describing a narrow
  method, or undersell a substantive result that never uses AGI buzzwords.
  `data/evals/judge_reliability_cases.json` holds cases built specifically
  to probe both directions, with a one-sided bound (`max_classification` /
  `min_classification`) rather than an exact match — the point is "not
  fooled," not "reproduces one specific score."

  Self-consistency — does the same paper get (roughly) the same score on
  repeat judgment? A judge whose score swings between Low and High on
  identical input is not a usable eval signal regardless of whether any
  one run happens to look right. Real providers are not perfectly
  deterministic even at temperature 0, so this reports the score range
  across repeated calls rather than asserting exact equality.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from agiresearch.agents.evaluator import evaluate_paper
from agiresearch.config import settings
from agiresearch.domain.schemas import Paper, PaperMetadata
from agiresearch.llm.client import build_chat_model

BACKEND_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_CASES_PATH = BACKEND_ROOT / "data" / "evals" / "judge_reliability_cases.json"

_CLASSIFICATION_ORDER = {
    "Low AGI Potential": 0,
    "Medium AGI Potential": 1,
    "High AGI Potential": 2,
}

_SELF_CONSISTENCY_PAPER = Paper(
    id="SELF-CONSISTENCY-001",
    title="Meta-Learning for Rapid Adaptation Across Simulated Control Tasks",
    link="https://example.invalid/self-consistency-001",
    metadata=PaperMetadata(
        authors=["A. Researcher"],
        abstract=(
            "We train a meta-learning policy that adapts to novel simulated control "
            "tasks from a small number of trials, evaluating transfer across 20 held-out "
            "task variations with consistent gains over a non-meta-learned baseline."
        ),
        published_date="2026-01-01T00:00:00",
        categories=["cs.LG"],
        source="fixture",
    ),
)


def load_calibration_cases() -> list[dict]:
    return json.loads(CALIBRATION_CASES_PATH.read_text())["cases"]


def _paper_from_case(case: dict) -> Paper:
    return Paper(
        id=case["id"],
        title=case["title"],
        link=f"https://example.invalid/{case['id']}",
        metadata=PaperMetadata(
            authors=[],
            abstract=case["abstract"],
            published_date="2026-01-01T00:00:00",
            categories=[],
            source="fixture",
        ),
    )


def run_calibration_eval(llm=None) -> dict:
    print("Calibration: judge vs. hand-built hype/substance probes")

    llm = llm or build_chat_model()
    cases = load_calibration_cases()
    passed = 0
    failures: list[str] = []

    for case in cases:
        paper = _paper_from_case(case)
        evaluation = evaluate_paper(paper, llm=llm)
        actual_rank = _CLASSIFICATION_ORDER[evaluation.classification]

        ok = True
        bound_desc = ""
        if "max_classification" in case:
            max_rank = _CLASSIFICATION_ORDER[case["max_classification"]]
            ok = actual_rank <= max_rank
            bound_desc = f"<= {case['max_classification']}"
        if "min_classification" in case:
            min_rank = _CLASSIFICATION_ORDER[case["min_classification"]]
            ok = ok and actual_rank >= min_rank
            bound_desc = f">= {case['min_classification']}"

        passed += ok
        marker = "PASS" if ok else "FAIL"
        print(
            f"  [{marker}] {case['id']}: expected {bound_desc}, "
            f"actual={evaluation.classification} (score={evaluation.agi_score})"
        )
        if not ok:
            failures.append(case["id"])
            print(f"           note: {case['note']}")

    print(f"\n  calibration: {passed}/{len(cases)} passed\n")
    return {"total": len(cases), "passed": passed, "failures": failures}


def run_self_consistency_eval(repeats: int = 3, llm=None) -> dict:
    print(f"Self-consistency: same paper judged {repeats}x")

    llm = llm or build_chat_model()
    scores = []
    for i in range(repeats):
        evaluation = evaluate_paper(_SELF_CONSISTENCY_PAPER, llm=llm)
        scores.append(evaluation.agi_score)
        print(f"  run {i + 1}: score={evaluation.agi_score} ({evaluation.classification})")

    score_range = max(scores) - min(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"\n  scores: {scores}")
    print(f"  range={score_range:.1f}  stdev={stdev:.2f}\n")

    return {"scores": scores, "range": score_range, "stdev": stdev}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "anthropic", "fake"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.provider:
        settings.llm_provider = args.provider  # type: ignore[assignment]
    if args.model:
        settings.llm_model = args.model

    if settings.llm_provider == "fake":
        print(
            "NOTE: the fake judge is a keyword heuristic, not a real judgment — both\n"
            "checks below are structural smoke tests in fake mode, not meaningful\n"
            "reliability signal. Pass --provider openai for a real run.\n"
        )

    llm = build_chat_model()
    run_calibration_eval(llm=llm)
    print("=" * 70)
    run_self_consistency_eval(repeats=args.repeats, llm=llm)


if __name__ == "__main__":
    main()
