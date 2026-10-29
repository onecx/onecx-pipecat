"""Permission: the user's OneCX voice permission and per-tool policy.

Two distinct concerns:

* ``PermissionAdapter`` asks onecx-permission-svc whether the user already
  holds the OneCX voice permission, through the same v1 API the OneCX
  permissions extension uses (``POST /v1/permissions/user/{productName}/{appId}``).
* ``resolve_tool_permission`` / ``allowed_tools`` apply the per-tool policy
  ai-provider already resolved into the pinned agent configuration; only
  ``ALWAYS_ALLOW`` tools are advertised.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from onecx_pipecat.config import PermissionSettings
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    AgentToolSnapshot,
    PermissionAction,
    Principal,
)


class PermissionAdapter(Protocol):
    async def has_voice_permission(
        self, principal: Principal, apm_principal_token: str, authorization: str
    ) -> bool:
        """Whether the user already has the OneCX voice permission."""
        ...


class HttpPermissionAdapter:
    """Check the voice permission via ``POST /v1/permissions/user/{productName}/{appId}``.

    The request matches the OneCX permissions Quarkus extension: the body
    ``token`` is the user's ``Authorization`` value (the access token) and
    ``apm-principal-token`` is propagated as a header. permission-svc reads the
    user's roles from the access token (by default ``realm_access/roles``,
    which Keycloak does not put into the ID token) and returns the
    application's permissions as ``{resource: [actions]}``. A missing access
    token or any error fails closed (no voice permission).
    """

    def __init__(
        self, settings: PermissionSettings | None = None, service_auth: httpx.Auth | None = None
    ) -> None:
        self._settings = settings or PermissionSettings()
        self._service_auth = service_auth
        self._client: httpx.AsyncClient | None = None

    def set_client(self, client: httpx.AsyncClient) -> None:
        """Inject a client (used by tests to avoid a live network call)."""
        self._client = client

    async def has_voice_permission(
        self, principal: Principal, apm_principal_token: str, authorization: str
    ) -> bool:
        if not authorization:
            return False
        s = self._settings
        client = self._client or httpx.AsyncClient()
        try:
            response = await client.post(
                f"{s.base_url}/v1/permissions/user/{s.voice_product_name}/{s.voice_app_id}",
                headers={"apm-principal-token": apm_principal_token},
                auth=self._service_auth,
                json={"token": authorization},
            )
            if not response.is_success:
                return False
            permissions: dict[str, Any] = response.json().get("permissions") or {}
        except (httpx.HTTPError, ValueError, AttributeError):
            return False
        finally:
            if self._client is None:
                await client.aclose()
        return s.voice_action in (permissions.get(s.voice_resource) or [])


def resolve_tool_permission(
    snapshot: AgentConfigSnapshot, tool: AgentToolSnapshot
) -> PermissionAction:
    """The tool rule of the tool's MCP server, else that server's unlisted policy.

    A tool from a server that is not in the snapshot is denied.
    """
    server = next((s for s in snapshot.mcp_servers if s.name == tool.server_name), None)
    if server is None:
        return PermissionAction.DENY
    for rule in server.tool_rules:
        if rule.tool_name == tool.tool_name:
            return rule.permission
    return server.unlisted_tool_permission


def allowed_tools(
    snapshot: AgentConfigSnapshot, discovered: list[AgentToolSnapshot]
) -> list[AgentToolSnapshot]:
    """Only tools resolved to ``ALWAYS_ALLOW`` are advertised."""
    return [
        t
        for t in discovered
        if resolve_tool_permission(snapshot, t) is PermissionAction.ALWAYS_ALLOW
    ]
