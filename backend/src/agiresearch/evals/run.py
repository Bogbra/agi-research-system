"""Golden-case regression eval: does the AGI judge classify obviously
low/medium/high-potential papers into the right band?

This eval's fake-mode run is *not* a meaningful pass/fail gate: the judge
reads free-text abstracts, and there's no deterministic signal to hang an
offline heuristic on that would actually mean anything — so the keyword
heuristic in `llm/fake.py` is explicitly not tuned to pass this eval, the
same way `evals/judge_reliability.py`'s calibration only means something
against a real provider. Fake mode still runs end to end here, to prove
the harness works with no API key.

Usage:
    uv run python -m agiresearch.evals.run
    uv run python -m agiresearch.evals.run --provider openai --model gpt-4o-mini

Exits non-zero (only when `--provider` isn't fake) if any golden paper's
score falls outside its expected `[min_score, max_score]` band — a range,
not an exact expected value, since testing a probabilistic judge against
one precise number tests noise, not calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agiresearch.agents.evaluator import evaluate_paper
from agiresearch.config import settings
from agiresearch.domain.schemas import Paper, PaperMetadata
from agiresearch.evals.result_logging import write_eval_result
from agiresearch.llm.client import build_chat_model

FIXTURES_PATH = Path(__file__).resolve().parents[3] / "data" / "test_cases" / "golden_papers.json"


def load_golden_papers() -> list[dict]:
    return json.loads(FIXTURES_PATH.read_text())["papers"]


def _paper_from_fixture(fixture: dict) -> Paper:
    return Paper(
        id=fixture["id"],
        title=fixture["title"],
        link=f"https://example.invalid/{fixture['id']}",
        metadata=PaperMetadata(
            authors=fixture.get("authors", []),
            abstract=fixture["abstract"],
            published_date="2026-01-01T00:00:00",
            categories=fixture.get("categories", []),
            source="golden-fixture",
        ),
    )


def run_evals(provider: str | None, model: str | None) -> int:
    if provider:
        settings.llm_provider = provider  # type: ignore[assignment]
    if model:
        settings.llm_model = model

    if settings.llm_provider == "fake":
        print(
            "NOTE: the fake judge is a keyword heuristic over free text, not a real\n"
            "judgment — it is not expected to reproduce the right classification band.\n"
            "Pass --provider openai for a meaningful run.\n"
        )

    papers = load_golden_papers()
    llm = build_chat_model()
    failures = 0
    case_results: list[dict] = []

    print(f"Running {len(papers)} golden paper(s) against LLM_PROVIDER={settings.llm_provider}\n")

    for fixture in papers:
        paper = _paper_from_fixture(fixture)
        min_score, max_score = fixture["min_score"], fixture["max_score"]

        evaluation = evaluate_paper(paper, llm=llm)
        score = evaluation.agi_score or 0.0

        ok = min_score <= score <= max_score
        failures += 0 if ok else 1
        marker = "PASS" if ok else "FAIL"

        print(
            f"[{marker}] {fixture['id']}: expected=[{min_score}, {max_score}] "
            f"actual={score} ({evaluation.classification})"
        )
        if not ok:
            print(f"         note: {fixture['note']}")

        case_results.append(
            {
                "id": fixture["id"],
                "min_score": min_score,
                "max_score": max_score,
                "actual_score": score,
                "classification": evaluation.classification,
                "passed": ok,
            }
        )

    print(f"\n{len(papers) - failures}/{len(papers)} golden papers passed.")

    if settings.llm_provider == "fake":
        return 0

    scores = [c["actual_score"] for c in case_results]
    result_path = write_eval_result(
        "golden-case",
        {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "num_cases": len(papers),
            "passed": len(papers) - failures,
            "failed": failures,
            "cases": case_results,
            "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        },
    )
    print(f"Result logged to {result_path}")

    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "anthropic", "fake"], default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    sys.exit(run_evals(args.provider, args.model))


if __name__ == "__main__":
    main()
