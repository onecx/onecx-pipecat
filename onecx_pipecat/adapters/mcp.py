"""MCP adapter: invoke a tool and surface the structured result.

Also discovers the session's tool surface once per session (``discover``): it
queries each MCP server of the pinned agent configuration a single time,
propagating the ``apm-principal-token``, and reports — rather than raises —
any server that fails, so the session continues in a degraded state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

from onecx_pipecat.config import McpSettings
from onecx_pipecat.models import AgentToolSnapshot, McpServerSnapshot, ToolProposal


@dataclass
class McpToolResult:
    """Structured result of a tool invocation (success or sanitized failure)."""

    ok: bool
    value: Any = None
    error: str = ""


@dataclass
class McpDiscovery:
    """The result of a one-time tool discovery across the agent's servers.

    ``degraded`` is a list of ``(server_name, reason)`` pairs: one entry per
    server that failed, so the session can emit a degraded-capability event
    and continue with the tools it was able to resolve.
    """

    tools: list[AgentToolSnapshot] = field(default_factory=list)
    degraded: list[tuple[str, str]] = field(default_factory=list)


def propagated_headers(apm_principal_token: str, authorization: str = "") -> dict[str, str]:
    """The user headers forwarded to MCP servers, as onecx-ai-provider-runtime does.

    The user's ``Authorization`` travels as ``UserAuthorization`` so it never
    collides with a server's own ``Authorization``.
    """
    headers: dict[str, str] = {}
    if apm_principal_token.strip():
        headers["apm-principal-token"] = apm_principal_token
    if authorization.strip():
        headers["UserAuthorization"] = authorization
    return headers


class McpAdapter(Protocol):
    async def invoke(
        self,
        server: McpServerSnapshot,
        proposal: ToolProposal,
        headers: Mapping[str, str],
        args: dict[str, Any] | None = None,
    ) -> McpToolResult: ...

    async def discover(
        self, servers: list[McpServerSnapshot], headers: Mapping[str, str]
    ) -> McpDiscovery:
        """Discover tools once, across the given servers, skipping failures."""
        ...


class HttpMcpAdapter:
    """Talk to the agent's MCP servers over MCP Streamable HTTP (official ``mcp`` SDK).

    Each call opens a session (``initialize``), sending the propagated user
    headers and applying ``mcp.timeout_seconds``. A server that fails discovery
    is captured as a degraded entry so the remaining servers still complete.
    """

    def __init__(self, settings: McpSettings | None = None) -> None:
        self._settings = settings or McpSettings()
        self._transport: httpx.AsyncBaseTransport | None = None

    def set_transport(self, transport: httpx.AsyncBaseTransport) -> None:
        """Inject an HTTP transport (tests run an in-process MCP server)."""
        self._transport = transport

    @asynccontextmanager
    async def _session(
        self, server: McpServerSnapshot, headers: Mapping[str, str]
    ) -> AsyncIterator[ClientSession]:
        async with httpx.AsyncClient(
            headers=dict(headers),
            timeout=self._settings.timeout_seconds,
            # Never follow redirects: custom headers (the user's apm-principal-token)
            # would be forwarded to whichever host the server redirects to.
            follow_redirects=False,
            transport=self._transport,
        ) as http_client:
            async with streamable_http_client(server.url, http_client=http_client) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def discover(
        self, servers: list[McpServerSnapshot], headers: Mapping[str, str]
    ) -> McpDiscovery:
        discovery = McpDiscovery()
        for server in servers:
            try:
                tools = await self._list_tools(server, headers)
            except Exception as exc:  # noqa: BLE001 - any server failure degrades, never aborts
                discovery.degraded.append((server.name, _reason(exc)))
                continue
            discovery.tools.extend(tools)
        return discovery

    async def _list_tools(
        self, server: McpServerSnapshot, headers: Mapping[str, str]
    ) -> list[AgentToolSnapshot]:
        tools: list[AgentToolSnapshot] = []
        async with self._session(server, headers) as session:
            cursor: str | None = None
            while True:
                page = await session.list_tools(cursor)
                tools.extend(
                    AgentToolSnapshot(
                        tool_name=tool.name,
                        tool_description=tool.description or "",
                        input_schema=dict(tool.inputSchema or {}),
                        server_name=server.name,
                        server_url=server.url,
                        required_permission=_required_permission(tool.meta),
                    )
                    for tool in page.tools
                )
                cursor = page.nextCursor
                if not cursor:
                    return tools

    async def invoke(
        self,
        server: McpServerSnapshot,
        proposal: ToolProposal,
        headers: Mapping[str, str],
        args: dict[str, Any] | None = None,
    ) -> McpToolResult:
        arguments = dict(args if args is not None else proposal.arguments)
        try:
            async with self._session(server, headers) as session:
                result = await session.call_tool(proposal.tool_name, arguments)
        except Exception as exc:  # noqa: BLE001 - sanitized boundary
            return McpToolResult(ok=False, error=_reason(exc))
        text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        if result.isError:
            return McpToolResult(ok=False, error=text or "tool reported an error")
        value = result.structuredContent if result.structuredContent is not None else text
        return McpToolResult(ok=True, value=value)


def _reason(exc: BaseException) -> str:
    """The innermost cause, unwrapped from the SDK's task-group exception groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def _required_permission(meta: Mapping[str, Any] | None) -> str:
    """The OneCX permission a tool declares in MCP ``_meta.requiredPermission``."""
    value = (meta or {}).get("requiredPermission")
    return value.strip() if isinstance(value, str) else ""


def tool_set_hash(tools: list[AgentToolSnapshot]) -> str:
    """Deterministic SHA-256 over the advertised tool set, independent of discovery order."""
    canonical = sorted(
        (tool.model_dump(mode="json") for tool in tools),
        key=lambda entry: (entry["server_name"], entry["tool_name"]),
    )
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
