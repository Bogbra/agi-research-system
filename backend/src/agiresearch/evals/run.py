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
classification doesn't match its expected band.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agiresearch.agents.evaluator import evaluate_paper
from agiresearch.config import settings
from agiresearch.domain.schemas import Paper, PaperMetadata
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

    print(f"Running {len(papers)} golden paper(s) against LLM_PROVIDER={settings.llm_provider}\n")

    for fixture in papers:
        paper = _paper_from_fixture(fixture)
        expected = fixture["expected_classification"]

        evaluation = evaluate_paper(paper, llm=llm)
        actual = evaluation.classification

        ok = actual == expected
        failures += 0 if ok else 1
        marker = "PASS" if ok else "FAIL"

        print(
            f"[{marker}] {fixture['id']}: expected={expected!r} actual={actual!r} "
            f"score={evaluation.agi_score}"
        )
        if not ok:
            print(f"         note: {fixture['note']}")

    print(f"\n{len(papers) - failures}/{len(papers)} golden papers passed.")

    if settings.llm_provider == "fake":
        return 0
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "anthropic", "fake"], default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    sys.exit(run_evals(args.provider, args.model))


if __name__ == "__main__":
    main()
