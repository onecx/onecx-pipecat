"""Transport-neutral voice-session state machine.

``VoiceSession`` owns one authenticated session: it starts it (auth, chat,
permission, pinned configuration, credentials, LLM/STT/TTS, MCP discovery),
serves ``refresh_token`` / ``stop`` once running, enforces the MCP tool policy
and runs the one long-lived Pipecat pipeline (see :mod:`onecx_pipecat.pipeline.builder`)
over the client's WebSocket until stop, token expiry or disconnect.

Starting a session is the whole authentication story: the ``start_session``
message carries the user's OneCX tokens (never the URL), and every step fails
closed with a typed ``ErrorEvent``. Only tools resolved to ``ALWAYS_ALLOW`` are
exposed to the LLM.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from fastapi import WebSocket
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor

from onecx_pipecat.adapters.agent_config import AgentConfigError
from onecx_pipecat.adapters.auth import AuthenticationError
from onecx_pipecat.adapters.chat import Chat, ChatAccessError
from onecx_pipecat.adapters.mcp import McpToolResult, propagated_headers
from onecx_pipecat.adapters.permission import allowed_tools
from onecx_pipecat.adapters.persistence import PersistenceError
from onecx_pipecat.container import AppContainer
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    AgentToolSnapshot,
    Principal,
    ToolProposal,
    VoiceEntry,
)
from onecx_pipecat.pipeline.builder import build_voice_pipeline
from onecx_pipecat.pipeline.prompt import compose_system_prompt
from onecx_pipecat.pipeline.runner import DeferredCloseWebSocket, run_worker
from onecx_pipecat.pipeline.tools import build_tools_schema
from onecx_pipecat.protocol import (
    Command,
    DegradedCapability,
    ErrorEvent,
    Event,
    RefreshToken,
    SessionClosed,
    SessionReady,
    StartSession,
    StopSession,
)
from onecx_pipecat.services.credentials import CredentialError
from onecx_pipecat.services.llm import LlmConfigError, build_llm
from onecx_pipecat.services.speech import (
    UnsupportedLanguageError,
    build_stt_service,
    build_tts_service,
    build_vad_analyzer,
)

_NORMAL_CLOSURE = 1000


class SessionState(StrEnum):
    CREATED = "CREATED"
    READY = "READY"
    CLOSED = "CLOSED"


class VoiceSession:
    """One focused voice session for one authenticated principal.

    The session's identity and tokens are established by the first
    ``start_session`` message, so the session is constructed before (and
    independently of) authentication.
    """

    def __init__(self, session_id: str, container: AppContainer) -> None:
        self.session_id = session_id
        self.state = SessionState.CREATED
        self._container = container
        self._principal: Principal | None = None
        self._apm_principal_token: str = ""
        self._authorization: str = ""
        self._snapshot: AgentConfigSnapshot | None = None
        self._chat: Chat | None = None
        self._allowed_tools: list[AgentToolSnapshot] = []
        self._start_failed = False
        self._llm: FrameProcessor | None = None
        self._stt: FrameProcessor | None = None
        self._tts: FrameProcessor | None = None
        self._tools: ToolsSchema | None = None
        self._system_prompt = ""
        self._history: list[dict[str, str]] = []
        self._worker: PipelineWorker | None = None
        self._ended = False
        self._final_events: list[Event] = []
        self._close_code = _NORMAL_CLOSURE

    # -- session identity -------------------------------------------------

    @property
    def principal(self) -> Principal | None:
        return self._principal

    @property
    def snapshot(self) -> AgentConfigSnapshot | None:
        return self._snapshot

    @property
    def allowed_tools(self) -> list[AgentToolSnapshot]:
        return list(self._allowed_tools)

    @property
    def apm_principal_token(self) -> str:
        return self._apm_principal_token

    @property
    def start_failed(self) -> bool:
        """True when the session was closed because ``start_session`` was rejected."""
        return self._start_failed

    async def dispatch(self, command: Command) -> list[Event]:
        match command:
            case StartSession():
                return await self._start(
                    command.agent_id,
                    command.chat_id,
                    command.apm_principal_token,
                    command.authorization,
                )
            case RefreshToken():
                return await self._refresh(command.apm_principal_token, command.authorization)
            case StopSession():
                return await self._stop()
        return [ErrorEvent(code="unknown_command", message="unsupported command")]

    def check_expiry(self) -> list[Event]:
        """End a READY session whose tokens have expired; no-op otherwise."""
        if self.state is not SessionState.READY:
            return []
        if not self._container.auth.is_expired(self._apm_principal_token):
            return []
        self.state = SessionState.CLOSED
        return [
            ErrorEvent(code="token_expired", message="session tokens expired without refresh"),
            SessionClosed(reason="token_expired"),
        ]

    async def invoke_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> McpToolResult:
        """Invoke a tool, but only one the pinned configuration resolved to ALWAYS_ALLOW."""
        if self.state is not SessionState.READY or self._snapshot is None:
            return McpToolResult(ok=False, error="session is not active")
        permitted = any(
            tool.server_name == server_name and tool.tool_name == tool_name
            for tool in self._allowed_tools
        )
        server = next((s for s in self._snapshot.mcp_servers if s.name == server_name), None)
        if not permitted or server is None:
            return McpToolResult(ok=False, error=f"tool {tool_name!r} is not permitted")
        proposal = ToolProposal(
            proposal_id=uuid4().hex,
            tool_name=tool_name,
            server_name=server_name,
            arguments=dict(arguments or {}),
        )
        return await self._container.mcp.invoke(
            server, proposal, propagated_headers(self._apm_principal_token, self._authorization)
        )

    # -- pipeline ---------------------------------------------------------

    async def run_pipeline(self, websocket: WebSocket) -> None:
        """Run the session's Pipecat pipeline over ``websocket`` until it ends.

        Ends on ``stop``, token expiry (:meth:`end_pipeline`), disconnect or
        :meth:`release`. The final events are then sent and the socket closed.
        """
        if self._llm is None or self._stt is None or self._tts is None or self._tools is None:
            raise RuntimeError("session is not started")
        if not self._ended:
            self._worker = build_voice_pipeline(
                websocket=DeferredCloseWebSocket(websocket),
                session_id=self.session_id,
                settings=self._container.settings,
                system_prompt=self._system_prompt,
                tools=self._tools,
                llm=self._llm,
                stt=self._stt,
                tts=self._tts,
                history=self._history,
                save_entry=self._save_entry,
                dispatch=self.dispatch,
                vad_analyzer=build_vad_analyzer(self._container.settings),
            )
            try:
                await run_worker(self._worker)
            finally:
                self._worker = None
        self.state = SessionState.CLOSED
        try:
            for event in self._final_events or [SessionClosed(reason="pipeline_ended")]:
                await websocket.send_json(event.model_dump())
            await websocket.close(code=self._close_code)
        except Exception:  # noqa: BLE001 - the client may already be gone
            pass

    async def _save_entry(self, entry: VoiceEntry) -> str:
        """Append to the anchored chat with the current (refreshable) user token."""
        if self._chat is None or self._principal is None:
            raise PersistenceError("session has no chat")
        return await self._container.persistence.append(
            self._chat.chat_id,
            entry,
            apm_principal_token=self._apm_principal_token,
            user_id=self._principal.subject,
        )

    async def end_pipeline(self, events: list[Event], close_code: int) -> None:
        """Stop the pipeline; ``events`` are sent after it, then the socket closes."""
        self._final_events = events
        self._close_code = close_code
        self._ended = True
        if self._worker is not None:
            await self._worker.cancel(reason="session_ended")

    async def release(self) -> None:
        """Cancel the pipeline if it is still running; idempotent."""
        self._ended = True
        if self._worker is not None:
            await self._worker.cancel(reason="released")

    # -- start / refresh / stop -------------------------------------------

    def _reset(self) -> None:
        """Drop everything a half-finished start may have established."""
        self._principal = None
        self._apm_principal_token = ""
        self._authorization = ""
        self._snapshot = None
        self._chat = None
        self._history = []
        self._allowed_tools = []
        self._llm = self._stt = self._tts = self._tools = None

    def _fail_start(self, code: str, message: str) -> list[Event]:
        self._reset()
        self.state = SessionState.CLOSED
        self._start_failed = True
        return [ErrorEvent(code=code, message=message), SessionClosed(reason="start_failed")]

    async def _start(
        self,
        agent_id: str,
        chat_id: str,
        apm_principal_token: str,
        authorization: str,
    ) -> list[Event]:
        if self.state is not SessionState.CREATED:
            return [ErrorEvent(code="illegal_state", message="session already started")]

        # 1. Authenticate the token carried by the message (never the URL).
        try:
            principal = await self._container.auth.authenticate(apm_principal_token)
        except AuthenticationError as exc:
            return self._fail_start("authentication_failed", str(exc))
        self._principal = principal
        self._apm_principal_token = apm_principal_token
        self._authorization = authorization

        # 2. Load the chat; fail closed if missing/inaccessible or not the user's.
        try:
            chat = await self._container.chat.load(chat_id, apm_principal_token)
        except ChatAccessError as exc:
            return self._fail_start("chat_not_accessible", str(exc))
        if principal.subject not in chat.user_ids:
            return self._fail_start("chat_not_accessible", f"chat {chat_id!r} not accessible")
        self._chat = chat

        # 2b. Load the chat history to seed the LLM context; fail closed on error.
        try:
            self._history = await self._container.persistence.history(
                chat_id,
                apm_principal_token=apm_principal_token,
                limit=self._container.settings.chat.history_max_messages,
            )
        except PersistenceError as exc:
            return self._fail_start("chat_not_accessible", f"chat history unavailable: {exc}")

        # 3. Check the user's existing OneCX voice permission.
        if not await self._container.permission.has_voice_permission(
            principal, apm_principal_token, authorization
        ):
            return self._fail_start("voice_not_permitted", "user lacks the voice permission")

        # 4. Pin the agent configuration for the whole session.
        try:
            snapshot = await self._container.agent_config.pin(
                principal.tenant_id, agent_id, apm_principal_token
            )
        except AgentConfigError as exc:
            return self._fail_start("config_unavailable", str(exc))
        self._snapshot = snapshot

        # 5. Build the LLM and speech services for the pinned provider and language.
        settings = self._container.settings
        try:
            llm = build_llm(snapshot, settings, self._container.credentials)
            stt = build_stt_service(settings, snapshot.language_code)
            tts = build_tts_service(settings, snapshot.language_code)
        except (CredentialError, LlmConfigError) as exc:
            return self._fail_start("config_unavailable", str(exc))
        except UnsupportedLanguageError as exc:
            return self._fail_start("unsupported_language", str(exc))

        # 6. Discover tools once on the agent's own MCP servers, forwarding the user's tokens.
        discovery = await self._container.mcp.discover(
            snapshot.mcp_servers, propagated_headers(apm_principal_token, authorization)
        )

        # 7. Expose only tools the pinned configuration resolves to ALWAYS_ALLOW.
        self._allowed_tools = allowed_tools(snapshot, discovery.tools)
        self._llm, self._stt, self._tts = llm, stt, tts
        self._tools = build_tools_schema(llm, self, snapshot, self._allowed_tools)
        self._system_prompt = compose_system_prompt(snapshot)

        self.state = SessionState.READY
        events: list[Event] = [
            SessionReady(
                session_id=self.session_id,
                agent_id=snapshot.agent_id,
                schema_version=snapshot.schema_version,
            )
        ]
        # 8. Report each skipped MCP server; the session continues degraded.
        for server_name, reason in discovery.degraded:
            events.append(DegradedCapability(reason=f"{server_name}: {reason}"))
        return events

    async def _refresh(self, apm_principal_token: str, authorization: str) -> list[Event]:
        """Replace the session tokens; fail closed if the new token is invalid."""
        if self.state is not SessionState.READY:
            return [ErrorEvent(code="illegal_state", message="session is not active")]
        try:
            principal = await self._container.auth.refresh(apm_principal_token)
        except AuthenticationError as exc:
            return [ErrorEvent(code="authentication_failed", message=str(exc))]
        # A refresh extends the same user's session; it must never switch the user,
        # whose chat access and voice permission were checked only at start.
        current = self._principal
        if (
            current is None
            or principal.subject != current.subject
            or principal.tenant_id != current.tenant_id
        ):
            return [
                ErrorEvent(
                    code="principal_mismatch",
                    message="refreshed token belongs to a different user",
                )
            ]
        self._principal = principal
        self._apm_principal_token = apm_principal_token
        self._authorization = authorization
        return []

    async def _stop(self) -> list[Event]:
        events: list[Event] = [SessionClosed(reason="client_stopped")]
        self.state = SessionState.CLOSED
        self._final_events = events
        self._close_code = _NORMAL_CLOSURE
        self._ended = True
        return events
