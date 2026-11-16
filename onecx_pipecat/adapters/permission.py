"""Permission: the user's OneCX permissions and per-tool policy.

Two distinct concerns:

* ``PermissionAdapter`` asks onecx-permission-svc for the user's permissions
  through the same v1 API the OneCX permissions extension uses
  (``POST /v1/permissions/user/{productName}/{appId}``). It answers the voice
  permission and, per turn, the permissions that gate MCP tools.
* ``resolve_tool_permission`` / ``allowed_tools`` apply the per-tool policy
  ai-provider already resolved into the pinned agent configuration (only
  ``ALWAYS_ALLOW`` tools are advertised) and the tool's OneCX
  ``requiredPermission``. A failed lookup fails closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from urllib.parse import quote

import httpx

from onecx_pipecat.config import PermissionSettings
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    AgentToolSnapshot,
    McpServerSnapshot,
    PermissionAction,
    Principal,
)


class PermissionAdapter(Protocol):
    async def has_voice_permission(
        self, principal: Principal, apm_principal_token: str, authorization: str
    ) -> bool:
        """Whether the user already has the OneCX voice permission."""
        ...

    async def application_permissions(
        self, product_name: str, app_id: str, apm_principal_token: str, authorization: str
    ) -> dict[str, list[str]] | None:
        """The user's ``{resource: [actions]}`` for one application; ``None`` when unknown."""
        ...


class HttpPermissionAdapter:
    """Check permissions via ``POST /v1/permissions/user/{productName}/{appId}``.

    The request matches the OneCX permissions Quarkus extension: the body
    ``token`` is the user's ``Authorization`` value (the access token) and
    ``apm-principal-token`` is propagated as a header. permission-svc reads the
    user's roles from the access token (by default ``realm_access/roles``,
    which Keycloak does not put into the ID token) and returns the
    application's permissions as ``{resource: [actions]}``. A missing access
    token or any error fails closed (no permissions).
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
        s = self._settings
        permissions = await self.application_permissions(
            s.voice_product_name, s.voice_app_id, apm_principal_token, authorization
        )
        if permissions is None:
            return False
        return s.voice_action in (permissions.get(s.voice_resource) or [])

    async def application_permissions(
        self, product_name: str, app_id: str, apm_principal_token: str, authorization: str
    ) -> dict[str, list[str]] | None:
        if not authorization:
            return None
        s = self._settings
        client = self._client or httpx.AsyncClient()
        try:
            response = await client.post(
                f"{s.base_url}/v1/permissions/user/"
                f"{quote(product_name, safe='')}/{quote(app_id, safe='')}",
                headers={"apm-principal-token": apm_principal_token},
                auth=self._service_auth,
                json={"token": authorization},
            )
            if not response.is_success:
                return None
            permissions = response.json().get("permissions") or {}
            if not isinstance(permissions, dict):
                return None
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        finally:
            if self._client is None:
                await client.aclose()
        return permissions


def parse_required_permission(value: str) -> tuple[str, str] | None:
    """Split ``onecx:<area>#<action>`` into ``(resource, action)``; ``None`` when malformed."""
    resource, separator, action = value.partition("#")
    if not separator or not resource.strip() or not action.strip():
        return None
    return resource.strip(), action.strip()


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
    snapshot: AgentConfigSnapshot,
    discovered: list[AgentToolSnapshot],
    granted: Mapping[tuple[str, str], Mapping[str, list[str]] | None] | None = None,
) -> list[AgentToolSnapshot]:
    """Tools resolved to ``ALWAYS_ALLOW`` whose OneCX permission the user holds.

    ``granted`` maps ``(productName, appId)`` to the user's permissions for that
    application (``None`` when the lookup failed). A tool that declares a
    ``requiredPermission`` is advertised only when the lookup succeeded and holds
    that permission; tools without one are not gated by this check.
    """
    grants = granted or {}
    servers = {server.name: server for server in snapshot.mcp_servers}
    return [
        tool
        for tool in discovered
        if resolve_tool_permission(snapshot, tool) is PermissionAction.ALWAYS_ALLOW
        and _holds_required_permission(tool, servers.get(tool.server_name), grants)
    ]


def required_permission_apps(
    snapshot: AgentConfigSnapshot, discovered: list[AgentToolSnapshot]
) -> set[tuple[str, str]]:
    """The ``(productName, appId)`` pairs whose permissions gate the discovered tools.

    Only apps that a discovered, policy-allowed tool actually needs are listed,
    so a turn performs the narrowest lookup it can.
    """
    servers = {server.name: server for server in snapshot.mcp_servers}
    apps: set[tuple[str, str]] = set()
    for tool in discovered:
        server = servers.get(tool.server_name)
        if not tool.required_permission or server is None:
            continue
        if resolve_tool_permission(snapshot, tool) is not PermissionAction.ALWAYS_ALLOW:
            continue
        if server.permission_product_name and server.permission_app_id:
            apps.add((server.permission_product_name, server.permission_app_id))
    return apps


def _holds_required_permission(
    tool: AgentToolSnapshot,
    server: McpServerSnapshot | None,
    granted: Mapping[tuple[str, str], Mapping[str, list[str]] | None],
) -> bool:
    if not tool.required_permission:
        return True
    if server is None or not server.permission_product_name or not server.permission_app_id:
        return False
    required = parse_required_permission(tool.required_permission)
    permissions = granted.get((server.permission_product_name, server.permission_app_id))
    if required is None or permissions is None:
        return False
    resource, action = required
    return action in (permissions.get(resource) or [])
