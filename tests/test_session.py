import pytest

from onecx_pipecat.config import Settings, SpeechSettings
from onecx_pipecat.container import AppContainer
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    McpServerSnapshot,
    ModelSnapshot,
    PermissionAction,
    Principal,
    ProviderSnapshot,
    ToolRule,
)
from onecx_pipecat.protocol import RefreshToken, StartSession, StopSession
from onecx_pipecat.session import SessionState, VoiceSession
from tests.fakes import FakeAgentConfig, fake_container


def _container() -> AppContainer:
    return fake_container()


def _start(**overrides) -> StartSession:
    kwargs = {"agent_id": "onecx-agent", "chat_id": "chat-1", "apm_principal_token": "tok-user"}
    kwargs.update(overrides)
    return StartSession(**kwargs)


def _session(container: AppContainer) -> VoiceSession:
    return VoiceSession("sess-1", container)


def _assert_start_rejected(session: VoiceSession, events: list, code: str) -> None:
    """A failed start closes the session and drops any half-established state."""
    assert [e.kind for e in events] == ["error", "session_closed"]
    assert events[0].code == code
    assert session.state is SessionState.CLOSED
    assert session.start_failed
    assert session.principal is None
    assert session.snapshot is None
    assert session.apm_principal_token == ""


async def test_start_becomes_ready_with_pinned_configuration():
    session = _session(_container())
    events = await session.dispatch(_start())
    assert session.state is SessionState.READY
    assert [e.kind for e in events] == ["session_ready"]
    assert events[0].schema_version == "1"
    assert session.snapshot is not None and session.snapshot.system_prompt == "be brief"


async def test_start_without_configuration_fails_closed():
    container = _container()
    container.agent_config = FakeAgentConfig()
    session = _session(container)
    events = await session.dispatch(_start())
    _assert_start_rejected(session, events, "config_unavailable")


async def test_start_with_unknown_token_fails_authentication():
    session = _session(_container())
    events = await session.dispatch(_start(apm_principal_token="nope"))
    _assert_start_rejected(session, events, "authentication_failed")


async def test_start_with_expired_token_fails_authentication():
    container = _container()
    container.auth.mark_expired("tok-user")
    session = _session(container)
    events = await session.dispatch(_start())
    _assert_start_rejected(session, events, "authentication_failed")


async def test_start_with_unknown_chat_fails():
    session = _session(_container())
    events = await session.dispatch(_start(chat_id="missing"))
    _assert_start_rejected(session, events, "chat_not_accessible")


async def test_start_in_a_chat_of_another_user_fails():
    container = _container()
    container.chat.register("chat-2", users=("someone-else",))
    session = _session(container)
    events = await session.dispatch(_start(chat_id="chat-2"))
    _assert_start_rejected(session, events, "chat_not_accessible")


async def test_start_without_voice_permission_fails():
    container = _container()
    container.permission.deny_voice("user-1")
    session = _session(container)
    events = await session.dispatch(_start())
    _assert_start_rejected(session, events, "voice_not_permitted")


async def test_start_checks_voice_permission_with_the_access_token():
    container = _container()
    session = _session(container)
    await session.dispatch(_start(authorization="Bearer access-user"))
    assert container.permission.authorizations == ["Bearer access-user"]


async def test_start_verifies_token_from_message_not_url():
    """The token is validated against the auth adapter, not the query string."""
    seen: list[str] = []

    class RecordingAuth:
        async def authenticate(self, token: str) -> Principal:
            seen.append(token)
            return Principal(subject="user-1", tenant_id="local")

        def is_expired(self, token: str) -> bool:
            return False

        async def refresh(self, token: str) -> Principal:
            return Principal(subject="user-1", tenant_id="local")

        async def service_token(self) -> str:
            return "svc"

    container = _container()
    container.auth = RecordingAuth()  # type: ignore[assignment]
    session = _session(container)
    await session.dispatch(_start(apm_principal_token="from-message"))
    assert seen == ["from-message"]


def _snapshot_with_servers() -> AgentConfigSnapshot:
    return AgentConfigSnapshot(
        agent_id="onecx-agent",
        schema_version="1",
        mcp_servers=[
            McpServerSnapshot(
                name="crm",
                url="https://crm.example/mcp",
                unlisted_tool_permission=PermissionAction.DENY,
                tool_rules=[
                    ToolRule(tool_name="safe", permission=PermissionAction.ALWAYS_ALLOW),
                    ToolRule(tool_name="asked", permission=PermissionAction.ALWAYS_ASK),
                ],
            ),
            McpServerSnapshot(
                name="docs",
                url="https://docs.example/mcp",
                unlisted_tool_permission=PermissionAction.ALWAYS_ALLOW,
            ),
        ],
    )


async def test_discovery_uses_the_agents_servers_and_forwards_user_tokens():
    """MCP servers come from the pinned agent configuration, not global settings."""
    container = _container()
    container.agent_config.register(_snapshot_with_servers())
    session = _session(container)
    await session.dispatch(_start(authorization="Bearer user"))
    assert container.mcp.calls == [
        (["crm", "docs"], {"apm-principal-token": "tok-user", "UserAuthorization": "Bearer user"})
    ]


async def test_start_filters_to_always_allow_tools_only():
    container = _container()
    container.agent_config.register(_snapshot_with_servers())
    container.mcp.tools = {"crm": ["safe", "asked", "listed"], "docs": ["search"]}
    session = _session(container)
    await session.dispatch(_start())
    # crm: "safe" is ALWAYS_ALLOW, "asked" ALWAYS_ASK, "listed" falls to DENY;
    # docs: "search" has no rule and falls to that server's ALWAYS_ALLOW.
    assert [(t.server_name, t.tool_name) for t in session.allowed_tools] == [
        ("crm", "safe"),
        ("docs", "search"),
    ]


async def test_start_emits_degraded_capability_for_failed_server():
    container = _container()
    container.agent_config.register(_snapshot_with_servers())
    container.mcp.failures = {"crm": "connection refused"}
    session = _session(container)
    events = await session.dispatch(_start())
    kinds = [e.kind for e in events]
    assert kinds == ["session_ready", "degraded_capability"]
    assert "crm" in events[1].reason


async def test_refresh_replaces_session_tokens():
    container = _container()
    container.auth.seed("tok-new", subject="user-1", tenant_id="local")
    session = _session(container)
    await session.dispatch(_start())
    events = await session.dispatch(
        RefreshToken(apm_principal_token="tok-new", authorization="Bearer tok-new")
    )
    assert events == []
    assert session.apm_principal_token == "tok-new"


async def test_refresh_for_a_different_user_is_rejected():
    container = _container()
    container.auth.seed("tok-other", subject="user-2", tenant_id="local")
    container.auth.seed("tok-tenant", subject="user-1", tenant_id="other-tenant")
    session = _session(container)
    await session.dispatch(_start())
    for token in ("tok-other", "tok-tenant"):
        events = await session.dispatch(RefreshToken(apm_principal_token=token))
        assert events[0].code == "principal_mismatch"
    assert session.apm_principal_token == "tok-user"
    assert session.principal is not None and session.principal.subject == "user-1"


async def test_refresh_with_invalid_token_fails():
    session = _session(_container())
    await session.dispatch(_start())
    events = await session.dispatch(RefreshToken(apm_principal_token="bogus"))
    assert events[0].kind == "error"
    assert events[0].code == "authentication_failed"


async def test_refresh_before_start_is_rejected():
    session = _session(_container())
    events = await session.dispatch(RefreshToken(apm_principal_token="tok-user"))
    assert events[0].code == "illegal_state"


async def test_expiry_is_detected_without_audio():
    container = _container()
    session = _session(container)
    await session.dispatch(_start())
    assert session.check_expiry() == []
    container.auth.mark_expired("tok-user")
    events = session.check_expiry()
    assert [e.kind for e in events] == ["error", "session_closed"]
    assert session.state is SessionState.CLOSED


async def test_refresh_is_exempt_from_the_expiry_check():
    container = _container()
    container.auth.seed("tok-new", subject="user-1", tenant_id="local")
    session = _session(container)
    await session.dispatch(_start())
    container.auth.mark_expired("tok-user")
    events = await session.dispatch(RefreshToken(apm_principal_token="tok-new"))
    assert events == []
    assert session.state is SessionState.READY


async def test_only_always_allow_tools_can_be_invoked():
    container = _container()
    container.agent_config.register(_snapshot_with_servers())
    container.mcp.tools = {"crm": ["safe", "asked", "hidden"], "docs": ["read"]}
    session = _session(container)
    await session.dispatch(_start())

    allowed = await session.invoke_tool("docs", "read", {"q": "x"})
    assert allowed.ok
    for server, tool in [("crm", "asked"), ("crm", "hidden"), ("crm", "unknown"), ("nope", "x")]:
        denied = await session.invoke_tool(server, tool)
        assert not denied.ok
    assert [(s, t) for s, t, _ in container.mcp.invocations] == [("docs", "read")]
    assert container.mcp.invocations[0][2]["apm-principal-token"] == "tok-user"


async def test_tools_cannot_be_invoked_before_start():
    session = _session(_container())
    assert not (await session.invoke_tool("docs", "read")).ok


async def test_second_start_is_rejected():
    session = _session(_container())
    await session.dispatch(_start())
    events = await session.dispatch(_start())
    assert events[0].kind == "error"
    assert events[0].code == "illegal_state"


async def test_stop_closes_the_session():
    session = _session(_container())
    await session.dispatch(_start())
    events = await session.dispatch(StopSession())
    assert session.state is SessionState.CLOSED
    assert events[0].kind == "session_closed"


async def test_start_without_provider_fails_closed_when_not_overridden():
    container = _container()
    container.settings = Settings()
    session = _session(container)
    events = await session.dispatch(_start())
    _assert_start_rejected(session, events, "config_unavailable")


async def test_start_with_unresolvable_credential_fails_closed():
    container = _container()
    container.settings = Settings()
    container.agent_config.register(
        AgentConfigSnapshot(
            agent_id="onecx-agent",
            schema_version="1",
            provider=ProviderSnapshot(
                id="p",
                name="p",
                type="OPENAI",
                llm_url="",
                auth_mode="API_KEY",
                credential_ref="missing",
            ),
            model=ModelSnapshot(
                id="m", name="m", model_identifier="gpt", communication_mode="HTTP"
            ),
        )
    )
    session = _session(container)
    events = await session.dispatch(_start())
    _assert_start_rejected(session, events, "config_unavailable")


async def test_unsupported_language_fails_closed_at_start():
    container = _container()
    container.settings = Settings(
        llm=container.settings.llm,
        speech=SpeechSettings(
            stt_provider="aws",
            tts_provider="aws",
            default_language="en",
            aws_profiles={"en": {"transcribe_language": "en-US", "polly_voice": "Joanna"}},
        ),
    )
    container.agent_config.register(
        AgentConfigSnapshot(
            agent_id="onecx-agent", schema_version="1", system_prompt="x", language_code="fr"
        )
    )
    session = _session(container)
    events = await session.dispatch(_start())
    _assert_start_rejected(session, events, "unsupported_language")


async def test_release_without_a_pipeline_is_a_noop():
    session = _session(_container())
    await session.release()
    await session.release()


async def test_run_pipeline_requires_a_started_session():
    session = _session(_container())
    with pytest.raises(RuntimeError):
        await session.run_pipeline(None)  # type: ignore[arg-type]
