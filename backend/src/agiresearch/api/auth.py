"""Authentication: a single static bearer token.

No role split, unlike a domain with a privileged human-review action to
gate — this pipeline has exactly one kind of caller and exactly one kind
of action (submit/read a research run), so a `caller`/`reviewer` split
would be a fabricated distinction with nothing on the other side of it.
Swapping this for a real identity provider touches only this module: every
router depends on `Principal`, never on a raw token.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agiresearch.config import settings

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    role: str = "caller"


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> Principal:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token.")
    if not _constant_time_eq(credentials.credentials, settings.api_bearer_token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid bearer token.")
    return Principal()
