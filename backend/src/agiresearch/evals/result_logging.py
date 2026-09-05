"""Reproducible eval-result logging.

Every real-provider eval run writes a dated JSON snapshot to
`eval-results/` — provider, model, per-case scores, pass/fail, mean,
stdev where applicable — so results can be compared after a later model
or prompt change instead of relying on memory or a stale paragraph in an
ADR. Not used for fake-mode smoke runs: those numbers aren't meaningful
(see each eval script's own docstring), so logging them would only add
noise to the comparison history this exists to build.
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
