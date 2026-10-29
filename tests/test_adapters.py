import base64
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from onecx_pipecat.adapters import (
    HttpAgentConfigAdapter,
    HttpChatAdapter,
    HttpMcpAdapter,
    HttpPermissionAdapter,
    InMemoryPersistence,
    JwtAuthAdapter,
)
from onecx_pipecat.adapters import auth as auth_module
from onecx_pipecat.adapters.agent_config import AgentConfigError
from onecx_pipecat.adapters.auth import (
    AuthenticationError,
    ClientCredentialsAuth,
    ServiceTokenError,
)
from onecx_pipecat.adapters.chat import ChatAccessError
from onecx_pipecat.adapters.mcp import propagated_headers
from onecx_pipecat.adapters.permission import allowed_tools, resolve_tool_permission
from onecx_pipecat.config import (
    AgentConfigSettings,
    AuthSettings,
    ChatSettings,
    PermissionSettings,
    PersistenceSettings,
)
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    AgentToolSnapshot,
    McpServerSnapshot,
    PermissionAction,
    Principal,
    ToolProposal,
    ToolRule,
    VoiceEntry,
    VoiceEntryKind,
)

_PRINCIPAL = Principal(subject="user-1", tenant_id="local")


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_persistence_round_trip():
    store = InMemoryPersistence()
    entry = VoiceEntry(kind=VoiceEntryKind.HUMAN, text="hi", idempotency_key="k1")
    entry_id = await store.append("s1", entry)
    entries = await store.list_entries("s1")
    assert len(entries) == 1
    fetched = await store.get("s1", entry_id)
    assert fetched is not None and fetched.text == "hi"


async def test_persistence_evicts_oldest_entry_when_cap_exceeded():
    store = InMemoryPersistence(PersistenceSettings(max_entries_per_session=3))
    for i in range(5):
        entry = VoiceEntry(
            kind=VoiceEntryKind.HUMAN,
            text=f"m{i}",
            idempotency_key=f"k{i}",
        )
        await store.append("s1", entry)
    entries = await store.list_entries("s1")
    assert len(entries) == 3
    # The oldest two entries (m0, m1) are evicted; the newest three remain.
    assert [e.text for e in entries] == ["m2", "m3", "m4"]


async def test_persistence_eviction_is_per_session():
    store = InMemoryPersistence(PersistenceSettings(max_entries_per_session=2))
    for i in range(3):
        await store.append("s1", VoiceEntry(kind=VoiceEntryKind.HUMAN, text=f"a{i}"))
    await store.append("s2", VoiceEntry(kind=VoiceEntryKind.HUMAN, text="s2-0"))
    assert [e.text for e in await store.list_entries("s1")] == ["a1", "a2"]
    assert [e.text for e in await store.list_entries("s2")] == ["s2-0"]


# --- JWT/JWKS verification (real signature + issuer + expiry) -----------


def _rsa_jwks():
    """A throwaway RS256 keypair and its JWKS document (test-only, offline)."""
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = key.public_key().public_numbers()
    n = pub.n.to_bytes((pub.n.bit_length() + 7) // 8, "big")
    e = pub.e.to_bytes(pub.e.bit_length(), "big")
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": "test-key",
                "n": jwt.utils.base64url_encode(n).decode(),
                "e": jwt.utils.base64url_encode(e).decode(),
            }
        ]
    }
    return key, jwks


def _signed_token(key, *, issuer, audience, subject, exp):
    import jwt

    return jwt.encode(
        {"sub": subject, "tenant_id": "acme", "iss": issuer, "aud": audience, "exp": exp},
        key,
        algorithm="RS256",
    )


async def test_jwt_auth_verifies_signature_issuer_and_expiry():
    from onecx_pipecat.adapters.auth import JwtAuthAdapter
    from onecx_pipecat.config import AuthSettings

    key, jwks = _rsa_jwks()
    settings = AuthSettings(issuer="https://issuer.example", audience="onecx-pipecat")
    auth = JwtAuthAdapter(settings)
    auth.set_jwks(jwks)
    token = _signed_token(
        key,
        issuer=settings.issuer,
        audience=settings.audience,
        subject="user-1",
        exp=10_000_000_000,
    )
    p = await auth.authenticate(token)
    assert p.subject == "user-1"
    assert p.tenant_id == "acme"
    assert auth.is_expired(token) is False


async def test_jwt_auth_rejects_bad_signature():
    from onecx_pipecat.adapters.auth import JwtAuthAdapter
    from onecx_pipecat.config import AuthSettings

    key, jwks = _rsa_jwks()
    other, _ = _rsa_jwks()
    settings = AuthSettings(issuer="https://issuer.example", audience="onecx-pipecat")
    auth = JwtAuthAdapter(settings)
    auth.set_jwks(jwks)
    token = _signed_token(
        other,
        issuer=settings.issuer,
        audience=settings.audience,
        subject="user-1",
        exp=10_000_000_000,
    )
    with pytest.raises(AuthenticationError):
        await auth.authenticate(token)


async def test_jwt_auth_rejects_wrong_issuer_and_audience():
    from onecx_pipecat.adapters.auth import JwtAuthAdapter
    from onecx_pipecat.config import AuthSettings

    key, jwks = _rsa_jwks()
    settings = AuthSettings(issuer="https://issuer.example", audience="onecx-pipecat")
    auth = JwtAuthAdapter(settings)
    auth.set_jwks(jwks)
    bad_issuer = _signed_token(
        key,
        issuer="https://evil.example",
        audience="onecx-pipecat",
        subject="u",
        exp=10_000_000_000,
    )
    bad_aud = _signed_token(
        key,
        issuer="https://issuer.example",
        audience="someone-else",
        subject="u",
        exp=10_000_000_000,
    )
    with pytest.raises(AuthenticationError):
        await auth.authenticate(bad_issuer)
    with pytest.raises(AuthenticationError):
        await auth.authenticate(bad_aud)


async def test_jwt_auth_rejects_expired_token():
    from onecx_pipecat.adapters.auth import JwtAuthAdapter
    from onecx_pipecat.config import AuthSettings

    key, jwks = _rsa_jwks()
    settings = AuthSettings(issuer="https://issuer.example", audience="onecx-pipecat")
    auth = JwtAuthAdapter(settings)
    auth.set_jwks(jwks)
    token = _signed_token(
        key, issuer=settings.issuer, audience=settings.audience, subject="u", exp=1
    )
    with pytest.raises(AuthenticationError):
        await auth.authenticate(token)
    assert auth.is_expired(token) is True


def _rsa_jwk(kid: str):
    """A throwaway RS256 keypair and its JWK document entry (test-only)."""
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = key.public_key().public_numbers()
    n = pub.n.to_bytes((pub.n.bit_length() + 7) // 8, "big")
    e = pub.e.to_bytes(pub.e.bit_length(), "big")
    jwk = {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": jwt.utils.base64url_encode(n).decode(),
        "e": jwt.utils.base64url_encode(e).decode(),
    }
    return key, jwk


def _signed_token_with_kid(key, *, kid, issuer, audience, subject, exp):
    import jwt

    return jwt.encode(
        {"sub": subject, "tenant_id": "acme", "iss": issuer, "aud": audience, "exp": exp},
        key,
        algorithm="RS256",
        headers={"kid": kid},
    )


async def test_jwt_auth_matches_key_by_kid_across_rotation():
    # A JWKS holding several rotated keys: a token signed by the second key
    # must verify, which only works if the key is chosen by the token's kid
    # rather than always taking the first key in the set.
    from onecx_pipecat.adapters.auth import JwtAuthAdapter
    from onecx_pipecat.config import AuthSettings

    first_key, first_jwk = _rsa_jwk("key-v1")
    second_key, second_jwk = _rsa_jwk("key-v2")
    jwks = {"keys": [first_jwk, second_jwk]}
    settings = AuthSettings(issuer="https://issuer.example", audience="onecx-pipecat")
    auth = JwtAuthAdapter(settings)
    auth.set_jwks(jwks)

    second_token = _signed_token_with_kid(
        second_key,
        kid="key-v2",
        issuer=settings.issuer,
        audience=settings.audience,
        subject="user-2",
        exp=10_000_000_000,
    )
    p = await auth.authenticate(second_token)
    assert p.subject == "user-2"
    assert auth.is_expired(second_token) is False

    first_token = _signed_token_with_kid(
        first_key,
        kid="key-v1",
        issuer=settings.issuer,
        audience=settings.audience,
        subject="user-1",
        exp=10_000_000_000,
    )
    assert (await auth.authenticate(first_token)).subject == "user-1"


async def test_jwt_auth_rejects_token_with_unknown_kid():
    from onecx_pipecat.adapters.auth import JwtAuthAdapter
    from onecx_pipecat.config import AuthSettings

    key, jwk = _rsa_jwk("key-v1")
    settings = AuthSettings(issuer="https://issuer.example", audience="onecx-pipecat")
    auth = JwtAuthAdapter(settings)
    auth.set_jwks({"keys": [jwk]})
    # A token whose kid is not in the JWKS must be rejected rather than tried
    # against the wrong (first) key.
    orphan = _signed_token_with_kid(
        key,
        kid="key-v9",
        issuer=settings.issuer,
        audience=settings.audience,
        subject="u",
        exp=10_000_000_000,
    )
    with pytest.raises(AuthenticationError):
        await auth.authenticate(orphan)
    assert auth.is_expired(orphan) is True


async def test_jwt_auth_falls_back_to_first_key_without_kid():
    from onecx_pipecat.adapters.auth import JwtAuthAdapter
    from onecx_pipecat.config import AuthSettings

    key, jwks = _rsa_jwks()
    settings = AuthSettings(issuer="https://issuer.example", audience="onecx-pipecat")
    auth = JwtAuthAdapter(settings)
    auth.set_jwks(jwks)
    # _signed_token carries no kid; the single-key set still verifies it.
    token = _signed_token(
        key,
        issuer=settings.issuer,
        audience=settings.audience,
        subject="user-1",
        exp=10_000_000_000,
    )
    assert (await auth.authenticate(token)).subject == "user-1"


# --- agent configuration: ai-provider GET /v1/agents/{id}/configuration ----

_AGENT_CONFIGURATION = {
    "schemaVersion": "1",
    "context": {"tenantId": "local", "principal": "user-1"},
    "agent": {"id": "agent-1", "name": "Helper"},
    "voice": {"voiceEnabled": True, "languageCode": "en"},
    "scaffold": {"id": "s1", "name": "default", "systemPrompt": "be brief"},
    "provider": {
        "id": "p1",
        "name": "OpenAI",
        "type": "OPENAI",
        "llmUrl": "https://llm.example/v1",
        "authMode": "API_KEY",
        "credentialRef": "credential://provider/p1/api-key",
    },
    "model": {
        "id": "m1",
        "name": "gpt",
        "modelIdentifier": "gpt-4o-mini",
        "modelConfig": '{"temperature": 0.2}',
        "communicationMode": "CHAT",
    },
    "skills": [
        {
            "id": "k1",
            "name": "Refund",
            "description": "Refund policy",
            "instruction": "Check the order first.",
            "position": 1,
            "source": "TENANT",
        }
    ],
    "mcpServers": [
        {
            "id": "srv-1",
            "name": "crm",
            "url": "https://crm.example/mcp",
            "credentialRef": "crm-cred",
            "unlistedToolPermission": "ALWAYS_ASK",
            "unlistedToolPermissionSource": "TOOL_EXECUTION_POLICY",
            "toolRules": [
                {
                    "toolName": "read",
                    "description": "Read CRM",
                    "permission": "ALWAYS_ALLOW",
                    "policySource": "AGENT_TOOL_RULE",
                }
            ],
        }
    ],
}


async def _pin(status: int, body: dict | None = None, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=body)

    adapter = HttpAgentConfigAdapter(AgentConfigSettings(base_url="http://ai"))
    adapter.set_client(_client(handler))
    return await adapter.pin("local", "agent-1", "tok")


async def test_agent_config_parses_ai_provider_configuration():
    seen: list[httpx.Request] = []
    snap = await _pin(200, _AGENT_CONFIGURATION, seen)

    assert seen[0].method == "GET"
    assert str(seen[0].url) == "http://ai/v1/agents/agent-1/configuration"
    assert seen[0].headers["apm-principal-token"] == "tok"
    assert snap.agent_id == "agent-1"
    assert snap.schema_version == "1"
    assert snap.system_prompt == "be brief"
    assert snap.agent_name == "Helper"
    assert snap.provider.type == "OPENAI"
    assert snap.provider.llm_url == "https://llm.example/v1"
    assert snap.provider.auth_mode == "API_KEY"
    assert snap.provider.credential_ref == "credential://provider/p1/api-key"
    assert snap.model.model_identifier == "gpt-4o-mini"
    assert snap.model.model_params == '{"temperature": 0.2}'
    assert [(s.name, s.instruction, s.position, s.source) for s in snap.skills] == [
        ("Refund", "Check the order first.", 1, "TENANT")
    ]
    server = snap.mcp_servers[0]
    assert (server.id, server.name, server.url) == ("srv-1", "crm", "https://crm.example/mcp")
    assert server.credential_ref == "crm-cred"
    assert server.unlisted_tool_permission is PermissionAction.ALWAYS_ASK
    assert server.tool_rules == [
        ToolRule(tool_name="read", permission=PermissionAction.ALWAYS_ALLOW, description="Read CRM")
    ]


@pytest.mark.parametrize("voice", [{"voiceEnabled": False}, {}, None])
async def test_agent_config_without_voice_enabled_fails_closed(voice):
    body = {**_AGENT_CONFIGURATION, "voice": voice}
    with pytest.raises(AgentConfigError):
        await _pin(200, body)


@pytest.mark.parametrize("status", [404, 409, 500])
async def test_agent_config_error_status_fails(status):
    with pytest.raises(AgentConfigError):
        await _pin(status, {})


@pytest.mark.parametrize(
    "override",
    [
        {"provider": None},
        {"provider": {"id": "p1"}},
        {"provider": {"type": "BEDROCK"}},
        {"provider": "x"},
        {"model": None},
        {"model": {"modelIdentifier": "  "}},
        {"model": "x"},
        {"skills": "x"},
        {"skills": ["not-a-dict"]},
        {"skills": [{"name": "a", "position": "first"}]},
    ],
)
async def test_agent_config_missing_or_invalid_llm_fails_closed(override):
    with pytest.raises(AgentConfigError):
        await _pin(200, {**_AGENT_CONFIGURATION, **override})


async def test_agent_config_provider_type_is_case_insensitive_and_skills_optional():
    body = {**_AGENT_CONFIGURATION, "provider": {"type": "ollama"}}
    body.pop("skills")
    snap = await _pin(200, body)
    assert snap.provider.type == "OLLAMA"
    assert snap.skills == []


async def test_agent_config_server_without_url_is_invalid():
    body = {**_AGENT_CONFIGURATION, "mcpServers": [{"name": "crm"}]}
    with pytest.raises(AgentConfigError):
        await _pin(200, body)


@pytest.mark.parametrize(
    "body",
    [
        ["not", "a", "dict"],
        {**_AGENT_CONFIGURATION, "voice": "yes"},
        {**_AGENT_CONFIGURATION, "agent": ["x"]},
        {**_AGENT_CONFIGURATION, "scaffold": "x"},
        {**_AGENT_CONFIGURATION, "mcpServers": ["not-a-dict"]},
        {**_AGENT_CONFIGURATION, "mcpServers": 3},
    ],
)
async def test_agent_config_malformed_body_fails_closed(body):
    with pytest.raises(AgentConfigError):
        await _pin(200, body)


async def test_agent_config_non_json_body_fails_closed():
    adapter = HttpAgentConfigAdapter(AgentConfigSettings(base_url="http://ai"))
    adapter.set_client(_client(lambda _req: httpx.Response(200, content=b"<html>")))
    with pytest.raises(AgentConfigError):
        await adapter.pin("local", "agent-1", "tok")


async def test_agent_id_is_escaped_in_the_url_path():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_AGENT_CONFIGURATION)

    adapter = HttpAgentConfigAdapter(AgentConfigSettings(base_url="http://ai"))
    adapter.set_client(_client(handler))
    await adapter.pin("local", "a/b?c#d", "tok")
    assert seen[0].url.raw_path == b"/v1/agents/a%2Fb%3Fc%23d/configuration"


# --- chat: onecx-chat-svc GET /internal/chats/{id} -------------------------


async def test_chat_loads_from_internal_chat_endpoint():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "c1", "type": "AI_CHAT", "topic": "general"})

    chat = HttpChatAdapter(ChatSettings(base_url="http://chat"))
    chat.set_client(_client(handler))
    loaded = await chat.load("c1", "tok")

    assert str(seen[0].url) == "http://chat/internal/chats/c1"
    assert seen[0].headers["apm-principal-token"] == "tok"
    assert (loaded.chat_id, loaded.topic) == ("c1", "general")


@pytest.mark.parametrize("status", [403, 404, 500])
async def test_chat_error_status_raises_access_error(status):
    chat = HttpChatAdapter(ChatSettings(base_url="http://chat"))
    chat.set_client(_client(lambda _req: httpx.Response(status)))
    with pytest.raises(ChatAccessError):
        await chat.load("c1", "tok")


async def test_chat_exposes_owner_and_participants():
    body = {
        "id": "c1",
        "userId": "owner",
        "participants": [{"userId": "guest"}, {"userName": "no-id"}],
    }
    chat = HttpChatAdapter(ChatSettings(base_url="http://chat"))
    chat.set_client(_client(lambda _req: httpx.Response(200, json=body)))
    loaded = await chat.load("c1", "tok")
    assert loaded.user_ids == frozenset({"owner", "guest"})


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>"),
        httpx.Response(200, json=["not", "a", "dict"]),
        httpx.Response(200, json={"participants": "nope"}),
    ],
)
async def test_chat_malformed_body_raises_access_error(response):
    chat = HttpChatAdapter(ChatSettings(base_url="http://chat"))
    chat.set_client(_client(lambda _req: response))
    with pytest.raises(ChatAccessError):
        await chat.load("c1", "tok")


async def test_chat_id_is_escaped_in_the_url_path():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "x"})

    chat = HttpChatAdapter(ChatSettings(base_url="http://chat"))
    chat.set_client(_client(handler))
    await chat.load("a/b?c#d", "tok")
    assert seen[0].url.raw_path == b"/internal/chats/a%2Fb%3Fc%23d"


async def test_chat_closes_its_own_client_on_error_paths(monkeypatch):
    clients: list[httpx.AsyncClient] = []
    real_client = httpx.AsyncClient

    def factory() -> httpx.AsyncClient:
        client = real_client(transport=httpx.MockTransport(lambda _req: httpx.Response(500)))
        clients.append(client)
        return client

    monkeypatch.setattr("onecx_pipecat.adapters.chat.httpx.AsyncClient", factory)
    with pytest.raises(ChatAccessError):
        await HttpChatAdapter(ChatSettings(base_url="http://chat")).load("c1", "tok")
    assert clients[0].is_closed


# --- service token: OIDC client credentials --------------------------------

_AUTH = AuthSettings(issuer="http://kc/realms/onecx", client_id="pipecat", client_secret="s3cr3t")


def _token_endpoint(seen: list, status: int = 200, body: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body or {"access_token": "svc-1", "expires_in": 300})

    return _client(handler)


async def test_service_token_uses_client_credentials_and_is_cached():
    seen: list[httpx.Request] = []
    service_auth = ClientCredentialsAuth(_AUTH, _token_endpoint(seen))
    assert await service_auth.token() == "svc-1"
    assert await service_auth.token() == "svc-1"

    assert len(seen) == 1
    assert str(seen[0].url) == "http://kc/realms/onecx/protocol/openid-connect/token"
    assert seen[0].content == b"grant_type=client_credentials"
    expected = "Basic " + base64.b64encode(b"pipecat:s3cr3t").decode()
    assert seen[0].headers["authorization"] == expected


async def test_service_token_is_sent_as_bearer_with_principal_token():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "c1", "topic": "general"})

    chat = HttpChatAdapter(
        ChatSettings(base_url="http://chat"), ClientCredentialsAuth(_AUTH, _token_endpoint([]))
    )
    chat.set_client(_client(handler))
    await chat.load("c1", "tok")
    assert seen[0].headers["authorization"] == "Bearer svc-1"
    assert seen[0].headers["apm-principal-token"] == "tok"


async def test_failed_service_token_fails_the_call_closed():
    chat = HttpChatAdapter(
        ChatSettings(base_url="http://chat"),
        ClientCredentialsAuth(_AUTH, _token_endpoint([], status=401, body={"error": "x"})),
    )
    chat.set_client(_client(lambda _req: httpx.Response(200, json={"id": "c1"})))
    with pytest.raises(ChatAccessError):
        await chat.load("c1", "tok")


async def test_token_response_without_access_token_is_an_error():
    service_auth = ClientCredentialsAuth(_AUTH, _token_endpoint([], body={"token_type": "x"}))
    with pytest.raises(ServiceTokenError):
        await service_auth.token()


# --- JWKS: fetched from the Keycloak issuer ---------------------------------


def _jwks_endpoint(seen: list, *jwks: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"keys": list(jwks)})

    return _client(handler)


def _issuer_token(key, kid: str) -> str:
    return _signed_token_with_kid(
        key, kid=kid, issuer=_AUTH.issuer, audience=_AUTH.audience, subject="u", exp=10_000_000_000
    )


async def test_jwks_is_fetched_from_the_issuer_once():
    key, jwk = _rsa_jwk("k1")
    seen: list[httpx.Request] = []
    auth = JwtAuthAdapter(_AUTH)
    auth.set_client(_jwks_endpoint(seen, jwk))

    assert (await auth.authenticate(_issuer_token(key, "k1"))).subject == "u"
    assert (await auth.authenticate(_issuer_token(key, "k1"))).subject == "u"
    assert [str(r.url) for r in seen] == ["http://kc/realms/onecx/protocol/openid-connect/certs"]


async def test_unknown_kid_refetches_the_jwks_for_key_rotation(monkeypatch):
    monkeypatch.setattr(auth_module, "_JWKS_MIN_REFETCH_SECONDS", 0)
    old_key, old_jwk = _rsa_jwk("old")
    new_key, new_jwk = _rsa_jwk("new")
    auth = JwtAuthAdapter(_AUTH)
    auth.set_client(_jwks_endpoint([], old_jwk))
    await auth.authenticate(_issuer_token(old_key, "old"))

    auth.set_client(_jwks_endpoint([], old_jwk, new_jwk))
    assert (await auth.authenticate(_issuer_token(new_key, "new"))).subject == "u"


async def test_unknown_kids_cannot_force_repeated_jwks_fetches():
    key, jwk = _rsa_jwk("k1")
    seen: list[httpx.Request] = []
    auth = JwtAuthAdapter(_AUTH)
    auth.set_client(_jwks_endpoint(seen, jwk))
    await auth.authenticate(_issuer_token(key, "k1"))
    for kid in ("bogus-1", "bogus-2"):
        with pytest.raises(AuthenticationError):
            await auth.authenticate(_issuer_token(key, kid))
    assert len(seen) == 1


async def test_unreachable_jwks_rejects_the_token():
    key, _ = _rsa_jwk("k1")
    auth = JwtAuthAdapter(_AUTH)
    auth.set_client(_client(lambda _req: httpx.Response(503)))
    with pytest.raises(AuthenticationError):
        await auth.authenticate(_issuer_token(key, "k1"))


# --- mcp: Streamable HTTP against the agent's servers ----------------------


class _Hosts(httpx.AsyncBaseTransport):
    """Route requests by host and record them."""

    def __init__(self, routes: dict[str, httpx.AsyncBaseTransport]) -> None:
        self._routes = routes
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self._routes[request.url.host].handle_async_request(request)


@asynccontextmanager
async def _mcp_servers():
    """An in-process MCP server on 127.0.0.1 plus an unreachable one; same task as the test."""
    server = FastMCP("crm", stateless_http=True, json_response=True)

    @server.tool(description="Read a CRM record")
    def read(record_id: str) -> str:
        return f"record {record_id}"

    @server.tool(description="Always fails")
    def fail() -> str:
        raise ValueError("boom")

    transport = _Hosts(
        {
            "127.0.0.1": httpx.ASGITransport(app=server.streamable_http_app()),
            "down.example": httpx.MockTransport(lambda _req: httpx.Response(503)),
        }
    )
    async with server.session_manager.run():
        yield transport


_CRM = McpServerSnapshot(name="crm", url="http://127.0.0.1:8000/mcp")
_DOWN = McpServerSnapshot(name="down", url="http://down.example/mcp")


def _mcp(transport) -> HttpMcpAdapter:
    mcp = HttpMcpAdapter()
    mcp.set_transport(transport)
    return mcp


_USER_HEADERS = propagated_headers("tok", "Bearer user")


async def test_propagated_headers_mirror_the_runtime():
    assert _USER_HEADERS == {"apm-principal-token": "tok", "UserAuthorization": "Bearer user"}
    assert propagated_headers("tok", "  ") == {"apm-principal-token": "tok"}


async def test_mcp_discovers_tools_and_reports_failed_servers():
    async with _mcp_servers() as transport:
        discovery = await _mcp(transport).discover([_CRM, _DOWN], _USER_HEADERS)

    assert [(t.server_name, t.tool_name, t.tool_description) for t in discovery.tools] == [
        ("crm", "read", "Read a CRM record"),
        ("crm", "fail", "Always fails"),
    ]
    assert [name for name, _ in discovery.degraded] == ["down"]
    assert "503" in discovery.degraded[0][1]
    for request in transport.requests:
        assert request.headers["apm-principal-token"] == "tok"
        assert request.headers["UserAuthorization"] == "Bearer user"
        assert "authorization" not in request.headers


async def test_mcp_invokes_a_tool():
    proposal = ToolProposal(proposal_id="p1", tool_name="read", arguments={"record_id": "42"})
    async with _mcp_servers() as transport:
        result = await _mcp(transport).invoke(_CRM, proposal, _USER_HEADERS)
    assert result.ok is True
    assert "record 42" in str(result.value)


async def test_mcp_tool_error_is_a_sanitized_failure():
    proposal = ToolProposal(proposal_id="p1", tool_name="fail")
    async with _mcp_servers() as transport:
        result = await _mcp(transport).invoke(_CRM, proposal, _USER_HEADERS)
    assert result.ok is False
    assert "boom" in result.error


async def test_mcp_unreachable_server_is_a_sanitized_failure():
    proposal = ToolProposal(proposal_id="p1", tool_name="read")
    async with _mcp_servers() as transport:
        result = await _mcp(transport).invoke(_DOWN, proposal, _USER_HEADERS)
    assert result.ok is False
    assert "503" in result.error


async def test_mcp_does_not_follow_redirects_with_the_user_token():
    evil_requests: list[httpx.Request] = []

    def evil(request: httpx.Request) -> httpx.Response:
        evil_requests.append(request)
        return httpx.Response(200, json={})

    transport = _Hosts(
        {
            "127.0.0.1": httpx.MockTransport(
                lambda _req: httpx.Response(307, headers={"location": "http://evil.example/mcp"})
            ),
            "evil.example": httpx.MockTransport(evil),
        }
    )
    discovery = await _mcp(transport).discover([_CRM], _USER_HEADERS)
    assert [name for name, _ in discovery.degraded] == ["crm"]
    assert evil_requests == []


# --- permission: onecx-permission v1 POST /v1/permissions/user/{product}/{app} ---


def _permission_adapter(status: int, body: dict | None = None, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=body)

    perm = HttpPermissionAdapter(PermissionSettings(base_url="http://perm"))
    perm.set_client(_client(handler))
    return perm


def _app_permissions(permissions: dict) -> dict:
    return {"appId": "onecx-ai-ui", "productName": "onecx-ai", "permissions": permissions}


async def test_voice_permission_found_in_application_permissions():
    seen: list[httpx.Request] = []
    body = _app_permissions({"CHAT": ["USE"], "VOICE": ["VIEW", "USE"]})
    perm = _permission_adapter(200, body, seen)
    assert await perm.has_voice_permission(_PRINCIPAL, "tok", "Bearer access") is True
    assert seen[0].method == "POST"
    assert str(seen[0].url) == "http://perm/v1/permissions/user/onecx-ai/onecx-ai-ui"
    assert seen[0].headers["apm-principal-token"] == "tok"
    assert json.loads(seen[0].content) == {"token": "Bearer access"}


async def test_voice_permission_without_access_token_is_denied():
    seen: list[httpx.Request] = []
    perm = _permission_adapter(200, _app_permissions({"VOICE": ["USE"]}), seen)
    assert await perm.has_voice_permission(_PRINCIPAL, "tok", "") is False
    assert seen == []


@pytest.mark.parametrize(
    "permissions", [{}, {"CHAT": ["USE"]}, {"VOICE": ["VIEW"]}, {"VOICE": None}]
)
async def test_missing_voice_permission_is_denied(permissions):
    perm = _permission_adapter(200, _app_permissions(permissions))
    assert await perm.has_voice_permission(_PRINCIPAL, "tok", "Bearer access") is False


@pytest.mark.parametrize("status", [400, 401, 500])
async def test_voice_permission_fails_closed_on_error_status(status):
    perm = _permission_adapter(status, {"errorCode": "x"})
    assert await perm.has_voice_permission(_PRINCIPAL, "tok", "Bearer access") is False


async def test_voice_permission_fails_closed_when_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    perm = HttpPermissionAdapter(PermissionSettings(base_url="http://perm"))
    perm.set_client(_client(handler))
    assert await perm.has_voice_permission(_PRINCIPAL, "tok", "Bearer access") is False


# --- tool policy: resolved per MCP server by ai-provider -------------------

_POLICY_SNAPSHOT = AgentConfigSnapshot(
    agent_id="a",
    schema_version="1",
    mcp_servers=[
        McpServerSnapshot(
            name="crm",
            url="https://crm.example/mcp",
            unlisted_tool_permission=PermissionAction.ALWAYS_ALLOW,
            tool_rules=[ToolRule(tool_name="write", permission=PermissionAction.DENY)],
        ),
    ],
)


def test_tool_rule_wins_over_unlisted_policy():
    tool = AgentToolSnapshot(tool_name="write", server_name="crm")
    assert resolve_tool_permission(_POLICY_SNAPSHOT, tool) is PermissionAction.DENY


def test_unruled_tool_uses_its_servers_unlisted_policy():
    tool = AgentToolSnapshot(tool_name="read", server_name="crm")
    assert resolve_tool_permission(_POLICY_SNAPSHOT, tool) is PermissionAction.ALWAYS_ALLOW


def test_tool_from_unknown_server_is_denied():
    tool = AgentToolSnapshot(tool_name="read", server_name="elsewhere")
    assert resolve_tool_permission(_POLICY_SNAPSHOT, tool) is PermissionAction.DENY


def test_allowed_tools_filters_to_always_allow_only():
    discovered = [
        AgentToolSnapshot(tool_name="read", server_name="crm"),
        AgentToolSnapshot(tool_name="write", server_name="crm"),
        AgentToolSnapshot(tool_name="read", server_name="elsewhere"),
    ]
    allowed = allowed_tools(_POLICY_SNAPSHOT, discovered)
    assert [(t.server_name, t.tool_name) for t in allowed] == [("crm", "read")]
