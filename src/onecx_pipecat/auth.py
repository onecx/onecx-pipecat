"""Fake-backed authentication for the WebSocket handshake.

This query-parameter check is the sole authentication mechanism for this
fake-backed slice and is invoked from
``src/onecx_pipecat/transport/websocket.py`` before ``websocket.accept()``
is called.
"""

from starlette.websockets import WebSocket


class AuthenticationError(Exception):
    """Raised when the WebSocket handshake cannot be authenticated."""


def authenticate(websocket: WebSocket) -> tuple[str, str]:
    """Extract ``tenant_id`` and ``principal_id`` from the handshake.

    Returns the (tenant_id, principal_id) pair, or raises
    ``AuthenticationError`` if either is missing or empty.
    """
    tenant_id = websocket.query_params.get("tenant_id", "")
    principal_id = websocket.query_params.get("principal_id", "")
    if not tenant_id or not principal_id:
        raise AuthenticationError(
            "tenant_id and principal_id query parameters are required"
        )
    return tenant_id, principal_id
