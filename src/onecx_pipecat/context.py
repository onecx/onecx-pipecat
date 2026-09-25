"""Immutable tenant/principal/session identity bound to a voice session.

A ``SessionContext`` is created once at session start and passed explicitly
into every adapter call so that tenant and principal propagation is
structural rather than ambient.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SessionContext:
    """Identity carried by a single voice session.

    The context is immutable and is the single source of truth for the
    tenant and principal acting on the session. Every adapter method
    receives this object as its first argument.
    """

    session_id: str
    tenant_id: str
    principal_id: str
    session_type: Literal["AI_CHAT"]

    def __post_init__(self) -> None:
        if self.session_type != "AI_CHAT":
            raise ValueError("session_type must be AI_CHAT")
        if not self.tenant_id:
            raise ValueError("tenant_id must not be empty")
        if not self.principal_id:
            raise ValueError("principal_id must not be empty")
