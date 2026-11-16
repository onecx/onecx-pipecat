"""Agent Configuration adapter: pin a snapshot for a run.

Session start fetches ai-provider ``GET /v1/agents/{id}/configuration`` and
pins the payload for the whole session; changes apply to the next session.
``pin`` fails (``AgentConfigError``) when the agent is missing (``404``),
invalid (``409``), unreachable, or ``voice.voiceEnabled`` is not true — a
voice session must not start without a usable configuration.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from onecx_pipecat.config import AgentConfigSettings
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    McpServerSnapshot,
    ModelSnapshot,
    PermissionAction,
    Principal,
    ProviderSnapshot,
    SkillSnapshot,
    ToolRule,
)

SUPPORTED_PROVIDER_TYPES = frozenset({"OPENAI", "OLLAMA", "ANTHROPIC"})


class AgentConfigError(Exception):
    """Raised when an agent/snapshot cannot be pinned for a voice session."""


class AgentConfigAdapter(Protocol):
    """Replaceable agent-configuration boundary."""

    async def pin(
        self, principal: Principal, agent_id: str, apm_principal_token: str
    ) -> AgentConfigSnapshot: ...


# Version of the ai-provider configuration contract this client understands; it is
# part of the cache key so a contract change never reuses an older payload.
CONFIGURATION_CONTRACT_VERSION = "1"

_CacheKey = tuple[str, str, str, str]


class HttpAgentConfigAdapter:
    """Pin the agent configuration from ai-provider ``GET /v1/agents/{id}/configuration``.

    Propagates ``apm-principal-token`` so the configuration is fetched as the
    user. Each fetch sends ``If-None-Match`` with the cached ETag, so an
    unchanged configuration is answered with ``304`` and the cached snapshot is
    reused. Loads are cached and single-flighted per tenant, principal subject,
    agent and contract version: concurrent pins share one request. A failed
    refresh raises ``AgentConfigError`` and leaves the cache untouched.
    Missing (404), invalid (409), or unreachable agents raise
    ``AgentConfigError``; an agent without ``voice.voiceEnabled`` is rejected
    the same way.
    """

    def __init__(
        self,
        settings: AgentConfigSettings | None = None,
        service_auth: httpx.Auth | None = None,
    ) -> None:
        self._settings = settings or AgentConfigSettings()
        self._service_auth = service_auth
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[_CacheKey, tuple[str, AgentConfigSnapshot]] = {}
        self._inflight: dict[_CacheKey, asyncio.Future[AgentConfigSnapshot]] = {}

    def set_client(self, client: httpx.AsyncClient) -> None:
        """Inject a client (used by tests to avoid a live network call)."""
        self._client = client

    async def pin(
        self, principal: Principal, agent_id: str, apm_principal_token: str
    ) -> AgentConfigSnapshot:
        key: _CacheKey = (
            principal.tenant_id,
            principal.subject,
            agent_id,
            CONFIGURATION_CONTRACT_VERSION,
        )
        pending = self._inflight.get(key)
        if pending is not None:
            return await asyncio.shield(pending)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AgentConfigSnapshot] = loop.create_future()
        self._inflight[key] = future
        try:
            snapshot = await self._fetch(key, agent_id, apm_principal_token)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved; waiters re-raise it
            raise
        else:
            future.set_result(snapshot)
            return snapshot
        finally:
            self._inflight.pop(key, None)

    async def _fetch(
        self, key: _CacheKey, agent_id: str, apm_principal_token: str
    ) -> AgentConfigSnapshot:
        cached = self._cache.get(key)
        headers = {"apm-principal-token": apm_principal_token}
        if cached is not None:
            headers["If-None-Match"] = cached[0]
        client = self._client or httpx.AsyncClient()
        try:
            try:
                response = await client.get(
                    f"{self._settings.base_url}/v1/agents/{quote(agent_id, safe='')}/configuration",
                    headers=headers,
                    auth=self._service_auth,
                )
            except httpx.HTTPError as exc:
                raise AgentConfigError(f"agent configuration unreachable: {exc}") from exc
            if response.status_code == 304:
                if cached is None:
                    raise AgentConfigError(f"unexpected 304 for agent={agent_id}")
                return cached[1]
            if response.status_code == 404:
                raise AgentConfigError(f"no configuration for agent={agent_id}")
            if response.status_code == 409:
                raise AgentConfigError(f"configuration for agent={agent_id} is invalid")
            if not response.is_success:
                raise AgentConfigError(f"agent configuration error: {response.status_code}")
            try:
                body = response.json()
            except ValueError as exc:
                raise AgentConfigError(
                    f"agent configuration for agent={agent_id} is not valid JSON"
                ) from exc
            snapshot = _parse_configuration(body, agent_id)
            etag = response.headers.get("etag")
            if etag:
                self._cache[key] = (etag, snapshot)
            else:
                self._cache.pop(key, None)
            return snapshot
        finally:
            if client is not self._client:
                await client.aclose()


def _section(body: dict[str, Any], key: str, agent_id: str) -> dict[str, Any]:
    value = body.get(key) or {}
    if not isinstance(value, dict):
        raise AgentConfigError(f"invalid {key!r} section for agent={agent_id}")
    return value


def _parse_configuration(body: Any, agent_id: str) -> AgentConfigSnapshot:
    """Turn an ``AgentConfiguration`` payload into a snapshot, enforcing voiceEnabled."""
    if not isinstance(body, dict):
        raise AgentConfigError(f"invalid configuration for agent={agent_id}")
    voice = _section(body, "voice", agent_id)
    if voice.get("voiceEnabled") is not True:
        raise AgentConfigError(f"voice disabled for agent={agent_id}")

    try:
        servers = [_parse_mcp_server(entry) for entry in body.get("mcpServers") or []]
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise AgentConfigError(f"invalid MCP server configuration for agent={agent_id}") from exc

    agent = _section(body, "agent", agent_id)
    scaffold = _section(body, "scaffold", agent_id)
    provider = _parse_provider(_section(body, "provider", agent_id), agent_id)
    model = _parse_model(_section(body, "model", agent_id), agent_id)
    try:
        skills = [_parse_skill(entry) for entry in _list(body.get("skills"))]
    except (ValueError, TypeError, AttributeError) as exc:
        raise AgentConfigError(f"invalid skill configuration for agent={agent_id}") from exc
    return AgentConfigSnapshot(
        agent_id=str(agent.get("id") or agent_id),
        agent_name=str(agent.get("name") or ""),
        schema_version=str(body.get("schemaVersion", "")),
        system_prompt=str(scaffold.get("systemPrompt") or ""),
        provider=provider,
        model=model,
        skills=skills,
        mcp_servers=servers,
        # Pinned voice language; absent/empty falls back to the configured default.
        language_code=str(voice.get("languageCode") or ""),
    )


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("expected a list")
    return value


def _parse_provider(section: dict[str, Any], agent_id: str) -> ProviderSnapshot:
    provider_type = str(section.get("type") or "").strip().upper()
    if provider_type not in SUPPORTED_PROVIDER_TYPES:
        raise AgentConfigError(f"unsupported or missing LLM provider type for agent={agent_id}")
    return ProviderSnapshot(
        id=str(section.get("id") or ""),
        name=str(section.get("name") or ""),
        type=provider_type,
        llm_url=str(section.get("llmUrl") or ""),
        auth_mode=str(section.get("authMode") or "").strip().upper(),
        credential_ref=str(section.get("credentialRef") or ""),
    )


def _parse_model(section: dict[str, Any], agent_id: str) -> ModelSnapshot:
    identifier = str(section.get("modelIdentifier") or "").strip()
    if not identifier:
        raise AgentConfigError(f"missing LLM model identifier for agent={agent_id}")
    return ModelSnapshot(
        id=str(section.get("id") or ""),
        name=str(section.get("name") or ""),
        model_identifier=identifier,
        model_params=str(section.get("modelConfig") or ""),
        communication_mode=str(section.get("communicationMode") or ""),
    )


def _parse_skill(entry: Any) -> SkillSnapshot:
    if not isinstance(entry, dict):
        raise TypeError("skill entry must be an object")
    position = entry.get("position")
    return SkillSnapshot(
        id=str(entry.get("id") or ""),
        name=str(entry.get("name") or ""),
        description=str(entry.get("description") or ""),
        instruction=str(entry.get("instruction") or ""),
        position=int(position) if position is not None else 0,
        source=str(entry.get("source") or ""),
    )


def _parse_mcp_server(entry: dict[str, Any]) -> McpServerSnapshot:
    return McpServerSnapshot(
        id=str(entry.get("id") or ""),
        name=str(entry["name"]),
        url=str(entry["url"]),
        credential_ref=str(entry.get("credentialRef") or ""),
        permission_product_name=str(entry.get("permissionProductName") or ""),
        permission_app_id=str(entry.get("permissionAppId") or ""),
        # A server without a resolved unlisted policy advertises nothing it did not list.
        unlisted_tool_permission=PermissionAction(entry.get("unlistedToolPermission") or "DENY"),
        tool_rules=[
            ToolRule(
                tool_name=str(rule["toolName"]),
                permission=PermissionAction(rule["permission"]),
                description=str(rule.get("description") or ""),
            )
            for rule in entry.get("toolRules") or []
        ],
    )
