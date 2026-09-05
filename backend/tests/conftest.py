"""Session-wide test environment.

Must run before `agiresearch.config` is imported by any test module, so
these env vars are set at conftest *module* scope (executed once, at
collection start) rather than inside a fixture function.

LLM_PROVIDER and ARXIV_PROVIDER are forced to "fake" — not defaulted —
because pydantic-settings resolves real process env vars *and* `.env` file
values before falling back to the field default, and a developer's local
`backend/.env` legitimately sets `LLM_PROVIDER=openai` / `ARXIV_PROVIDER=live`
to run a live eval. Without this override, the test suite would silently
start making real, billed API calls and real network requests the moment
that `.env` file exists.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_tmp_dir = Path(tempfile.mkdtemp(prefix="agiresearch-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_dir / 'test.db'}")
os.environ["LLM_PROVIDER"] = "fake"
os.environ["ARXIV_PROVIDER"] = "fake"
