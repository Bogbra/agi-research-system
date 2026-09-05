"""Reproducible eval-result logging.

Every real-provider eval run writes a dated JSON snapshot to
`eval-results/` — provider, model, `prompt_sha256`
(`agents/evaluator.py:PROMPT_SHA256`), `rubric_version`
(`domain/scoring.py:RUBRIC_VERSION`), per-case scores, pass/fail, mean,
stdev where applicable. The three identifying fields (provider, prompt
hash, rubric version) exist specifically so that if a later run's numbers
differ, it's possible to tell *why* — a different model, a changed
prompt, or a changed rubric — instead of an unexplained drift. Not used
for fake-mode smoke runs: those numbers aren't meaningful (see each eval
script's own docstring), so logging them would only add noise to the
comparison history this exists to build.

Docs that cite these numbers (README, ADR 0006) name the exact file each
figure came from — a real-model eval result is a snapshot of one run, not
a permanent property of the system, and is expected to drift as the
model, prompt, or rubric changes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval-results"


def write_eval_result(eval_name: str, payload: dict) -> Path:
    """Write `payload` (already containing provider/model/case data) to a
    timestamped file under `eval-results/`. Returns the path written.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC)
    filename = f"{eval_name}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    path = RESULTS_DIR / filename
    record = {"timestamp": timestamp.isoformat(), **payload}
    path.write_text(json.dumps(record, indent=2))
    return path
