"""Turn-boundary pinning: conditional config refresh, tool-set hashing, the permission gate."""

import asyncio

import httpx
import pytest
from pipecat.frames.frames import (
    LLMUpdateSettingsFrame,
    STTUpdateSettingsFrame,
    TTSUpdateSettingsFrame,
)

from onecx_pipecat.adapters.agent_config import AgentConfigError, HttpAgentConfigAdapter
from onecx_pipecat.adapters.mcp import tool_set_hash
from onecx_pipecat.adapters.permission import (
    allowed_tools,
    parse_required_permission,
    required_permission_apps,
)
from onecx_pipecat.config import AgentConfigSettings, AwsSpeechProfile
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    AgentToolSnapshot,
    McpServerSnapshot,
    ModelSnapshot,
    PermissionAction,
    Principal,
    ProviderSnapshot,
)
from onecx_pipecat.protocol import DegradedCapability, StartSession
from onecx_pipecat.session import SessionState, VoiceSession
from tests.fakes import fake_container
from tests.test_adapters import _AGENT_CONFIGURATION

PRINCIPAL = Principal(subject="user-1", tenant_id="local")
OTHER_PRINCIPAL = Principal(subject="user-2", tenant_id="local")


def _adapter(handler) -> HttpAgentConfigAdapter:
    adapter = HttpAgentConfigAdapter(AgentConfigSettings(base_url="http://ai"))
    adapter.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return adapter


# --- conditional refresh and single-flight ---------------------------------


async def test_refresh_sends_if_none_match_and_reuses_cached_snapshot_on_304():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("If-None-Match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, json=_AGENT_CONFIGURATION, headers={"ETag": '"v1"'})

    adapter = _adapter(handler)
    first = await adapter.pin(PRINCIPAL, "agent-1", "tok")
    second = await adapter.pin(PRINCIPAL, "agent-1", "tok")

    assert "If-None-Match" not in seen[0].headers
    assert seen[1].headers["If-None-Match"] == '"v1"'
    assert second is first


async def test_changed_configuration_replaces_the_cached_version():
    etags = iter(['"v1"', '"v2"'])
    changed = {
        **_AGENT_CONFIGURATION,
        "scaffold": {**_AGENT_CONFIGURATION["scaffold"], "systemPrompt": "new"},
    }
    bodies = iter([_AGENT_CONFIGURATION, changed])

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(bodies), headers={"ETag": next(etags)})

    adapter = _adapter(handler)
    first = await adapter.pin(PRINCIPAL, "agent-1", "tok")
    second = await adapter.pin(PRINCIPAL, "agent-1", "tok")
    assert second.system_prompt == "new"
    assert second is not first


async def test_concurrent_pins_share_one_request():
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return httpx.Response(200, json=_AGENT_CONFIGURATION, headers={"ETag": '"v1"'})

    adapter = _adapter(handler)
    snapshots = await asyncio.gather(*(adapter.pin(PRINCIPAL, "agent-1", "tok") for _ in range(5)))
    assert calls == 1
    assert all(snapshot is snapshots[0] for snapshot in snapshots)


async def test_cache_is_isolated_by_principal_and_agent():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_AGENT_CONFIGURATION, headers={"ETag": '"v1"'})

    adapter = _adapter(handler)
    await adapter.pin(PRINCIPAL, "agent-1", "tok")
    await adapter.pin(OTHER_PRINCIPAL, "agent-1", "tok")
    await adapter.pin(PRINCIPAL, "agent-2", "tok")
    # Different principal or agent is a different cache key: no If-None-Match is sent.
    assert [request.headers.get("If-None-Match") for request in seen] == [None, None, None]


async def test_failed_refresh_raises_and_keeps_the_cache_usable():
    state = {"fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            return httpx.Response(503)
        if request.headers.get("If-None-Match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, json=_AGENT_CONFIGURATION, headers={"ETag": '"v1"'})

    adapter = _adapter(handler)
    pinned = await adapter.pin(PRINCIPAL, "agent-1", "tok")
    state["fail"] = True
    with pytest.raises(AgentConfigError):
        await adapter.pin(PRINCIPAL, "agent-1", "tok")
    state["fail"] = False
    assert await adapter.pin(PRINCIPAL, "agent-1", "tok") is pinned


# --- tool-set hashing -------------------------------------------------------


def _tool(name: str, server: str = "crm", required: str = "") -> AgentToolSnapshot:
    return AgentToolSnapshot(tool_name=name, server_name=server, required_permission=required)


def test_tool_set_hash_is_independent_of_discovery_order():
    tools = [_tool("a"), _tool("b", server="docs")]
    assert tool_set_hash(tools) == tool_set_hash(list(reversed(tools)))


def test_tool_set_hash_changes_with_the_advertised_tools():
    base = tool_set_hash([_tool("a"), _tool("b")])
    assert tool_set_hash([_tool("a")]) != base
    assert tool_set_hash([_tool("a"), _tool("c")]) != base


def test_tool_set_hash_changes_when_a_required_permission_changes():
    before = tool_set_hash([_tool("a", required="onecx:event#mcp-read")])
    after = tool_set_hash([_tool("a", required="onecx:event#mcp-write")])
    assert before != after


def test_empty_tool_set_has_a_stable_hash():
    assert tool_set_hash([]) == tool_set_hash([])


# --- permission gate --------------------------------------------------------


def _snapshot(*, permission_app: bool = True) -> AgentConfigSnapshot:
    return AgentConfigSnapshot(
        agent_id="agent-1",
        schema_version="1",
        mcp_servers=[
            McpServerSnapshot(
                name="events",
                url="https://events.example/mcp",
                unlisted_tool_permission=PermissionAction.ALWAYS_ALLOW,
                permission_product_name="onecx-event" if permission_app else "",
                permission_app_id="event-app" if permission_app else "",
            )
        ],
    )


def test_parse_required_permission_splits_resource_and_action():
    assert parse_required_permission("onecx:event#mcp-read") == ("onecx:event", "mcp-read")
    assert parse_required_permission("not-a-permission") is None
    assert parse_required_permission("") is None


def test_tool_is_advertised_only_when_the_user_holds_its_permission():
    discovered = [
        _tool("list", server="events", required="onecx:event#mcp-read"),
        _tool("delete", server="events", required="onecx:event#mcp-write"),
        _tool("ping", server="events"),
    ]
    granted = {("onecx-event", "event-app"): {"onecx:event": ["mcp-read"]}}
    allowed = allowed_tools(_snapshot(), discovered, granted)
    assert [t.tool_name for t in allowed] == ["list", "ping"]


def test_failed_permission_lookup_hides_gated_tools_only():
    discovered = [
        _tool("list", server="events", required="onecx:event#mcp-read"),
        _tool("ping", server="events"),
    ]
    allowed = allowed_tools(_snapshot(), discovered, {("onecx-event", "event-app"): None})
    assert [t.tool_name for t in allowed] == ["ping"]


def test_gated_tool_without_published_permission_app_is_hidden():
    discovered = [_tool("list", server="events", required="onecx:event#mcp-read")]
    allowed = allowed_tools(_snapshot(permission_app=False), discovered, {})
    assert allowed == []


def test_lookup_is_narrow_one_app_per_gated_tool_set():
    discovered = [
        _tool("list", server="events", required="onecx:event#mcp-read"),
        _tool("ping", server="events"),
    ]
    assert required_permission_apps(_snapshot(), discovered) == {("onecx-event", "event-app")}
    assert required_permission_apps(_snapshot(), [_tool("ping", server="events")]) == set()


# --- session: turn-boundary pinning -----------------------------------------


def _session_container(*, required: bool = True):
    container = fake_container()
    container.agent_config.register(_snapshot())
    container.mcp.tools = {"events": ["list", "ping"]}
    if required:
        container.mcp.required = {("events", "list"): "onecx:event#mcp-read"}
    container.permission.grants = {("onecx-event", "event-app"): {"onecx:event": ["mcp-read"]}}
    return container


async def _started(container) -> VoiceSession:
    session = VoiceSession("sess-1", container)
    await session.dispatch(
        StartSession(agent_id="agent-1", chat_id="chat-1", apm_principal_token="tok-user")
    )
    assert session.state is SessionState.READY
    return session


async def test_start_pins_first_turn_with_gated_tools_and_records_hash():
    container = _session_container()
    session = await _started(container)
    assert [t.tool_name for t in session.allowed_tools] == ["list", "ping"]
    assert session.turn_pin is not None
    assert session.turn_pin.turn_index == 1
    assert session.turn_pin.tool_set_hash == tool_set_hash(session.allowed_tools)
    assert container.permission.lookups == [("onecx-event", "event-app")]


async def test_configuration_and_tool_changes_apply_only_at_the_next_turn():
    container = _session_container()
    session = await _started(container)
    first_hash = session.turn_pin.tool_set_hash

    changed = _snapshot()
    changed.system_prompt = "updated"
    container.agent_config.register(changed)
    container.mcp.tools = {"events": ["ping", "new"]}

    # Mid-turn nothing changes: the pinned configuration and tool set stand.
    assert [t.tool_name for t in session.allowed_tools] == ["list", "ping"]
    assert session.snapshot.system_prompt == ""

    events = await session.begin_turn()
    assert events == []
    assert [t.tool_name for t in session.allowed_tools] == ["ping", "new"]
    assert session.snapshot.system_prompt == "updated"
    assert session.turn_pin.turn_index == 2
    assert session.turn_pin.tool_set_hash != first_hash


async def test_refresh_failure_keeps_the_pinned_configuration_and_tools():
    container = _session_container()
    session = await _started(container)
    before = [t.tool_name for t in session.allowed_tools]
    container.agent_config.fail_with = "ai-provider unavailable"
    container.mcp.tools = {"events": ["other"]}

    events = await session.begin_turn()

    assert session.state is SessionState.READY
    assert [t.tool_name for t in session.allowed_tools] == before
    assert len(events) == 1 and isinstance(events[0], DegradedCapability)
    assert "previous kept" in events[0].reason
    assert session.turn_pin.turn_index == 2


async def test_gate_is_reevaluated_each_turn():
    container = _session_container()
    session = await _started(container)
    container.permission.grants = {}
    await session.begin_turn()
    assert [t.tool_name for t in session.allowed_tools] == ["ping"]


async def test_begin_turn_is_a_noop_when_not_ready():
    session = VoiceSession("sess-1", _session_container())
    assert await session.begin_turn() == []
    assert session.turn_pin is None


# --- session: in-place model and language updates at the turn boundary ------


class _RecordingWorker:
    def __init__(self) -> None:
        self.frames: list = []

    async def queue_frame(self, frame) -> None:
        self.frames.append(frame)


class _SettingsService:
    """Stands in for a Pipecat service that exposes a ``Settings`` delta class."""

    class Settings:
        def __init__(self, **fields) -> None:
            self.fields = fields

    def register_function(self, name, handler) -> None:
        pass

    def unregister_function(self, name) -> None:
        pass


def _snapshot_with(
    *, provider: str = "OPENAI", model: str = "gpt-a", language: str = ""
) -> AgentConfigSnapshot:
    snapshot = _snapshot()
    snapshot.provider = ProviderSnapshot(type=provider, credential_ref="cred")
    snapshot.model = ModelSnapshot(model_identifier=model)
    snapshot.language_code = language
    return snapshot


async def _started_with_services(snapshot: AgentConfigSnapshot):
    container = _session_container()
    container.agent_config.register(snapshot)
    container.settings.speech.aws_profiles["pl"] = AwsSpeechProfile(
        transcribe_language="pl-PL", polly_voice="Ewa"
    )
    session = await _started(container)
    session._llm, session._stt, session._tts = (
        _SettingsService(),
        _SettingsService(),
        _SettingsService(),
    )
    session._worker = _RecordingWorker()
    return container, session


async def test_model_change_is_pushed_into_the_running_llm_at_the_turn_boundary():
    container, session = await _started_with_services(_snapshot_with(model="gpt-a"))
    container.agent_config.register(_snapshot_with(model="gpt-b"))

    assert await session.begin_turn() == []

    [frame] = session._worker.frames
    assert isinstance(frame, LLMUpdateSettingsFrame)
    assert frame.delta.fields == {"model": "gpt-b"}


async def test_provider_change_keeps_the_running_provider_and_reports_degraded():
    container, session = await _started_with_services(_snapshot_with(provider="OPENAI"))
    container.agent_config.register(_snapshot_with(provider="ANTHROPIC"))

    events = await session.begin_turn()

    assert session._worker.frames == []
    assert [type(e) for e in events] == [DegradedCapability]
    assert "next session" in events[0].reason


async def test_language_change_updates_stt_and_tts_when_the_profile_exists():
    container, session = await _started_with_services(_snapshot_with(language="en"))
    container.agent_config.register(_snapshot_with(language="pl"))

    assert await session.begin_turn() == []

    stt, tts = session._worker.frames
    assert isinstance(stt, STTUpdateSettingsFrame)
    assert stt.delta.fields == {"language": "pl-PL"}
    assert isinstance(tts, TTSUpdateSettingsFrame)
    assert tts.delta.fields == {"voice": "Ewa", "language": "pl-PL"}


async def test_unsupported_language_keeps_the_previous_language():
    container, session = await _started_with_services(_snapshot_with(language="en"))
    container.agent_config.register(_snapshot_with(language="xx"))

    events = await session.begin_turn()

    assert session._worker.frames == []
    assert [type(e) for e in events] == [DegradedCapability]
    assert "language change not applied" in events[0].reason
    assert session._running.language_code == "en"
