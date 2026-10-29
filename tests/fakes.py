"""Test doubles for the OneCX boundaries; production code ships only real adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

from onecx_pipecat.adapters.agent_config import AgentConfigError
from onecx_pipecat.adapters.auth import AuthenticationError
from onecx_pipecat.adapters.chat import ChatAccessError
from onecx_pipecat.adapters.mcp import McpDiscovery, McpToolResult
from onecx_pipecat.adapters.persistence import PersistenceError
from onecx_pipecat.config import LlmSettings, Settings, SpeechSettings
from onecx_pipecat.container import AppContainer
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    AgentToolSnapshot,
    McpServerSnapshot,
    Principal,
    ToolProposal,
    VoiceEntry,
)


class FakeAuth:
    def __init__(self) -> None:
        self._principals: dict[str, Principal] = {}
        self._expired: set[str] = set()

    def seed(
        self, token: str, subject: str, tenant_id: str, roles: list[str] | None = None
    ) -> None:
        self._principals[token] = Principal(
            subject=subject, tenant_id=tenant_id, roles=list(roles or [])
        )

    def mark_expired(self, token: str) -> None:
        self._expired.add(token)

    async def authenticate(self, token: str) -> Principal:
        if token not in self._principals or token in self._expired:
            raise AuthenticationError("unknown or expired token")
        return self._principals[token]

    def is_expired(self, token: str) -> bool:
        return token not in self._principals or token in self._expired

    async def refresh(self, token: str) -> Principal:
        return await self.authenticate(token)

    async def service_token(self) -> str:
        return "svc"


class FakeChat:
    def __init__(self) -> None:
        self._topics: dict[str, str] = {}
        self._users: dict[str, frozenset[str]] = {}

    def register(self, chat_id: str, topic: str = "", users: tuple[str, ...] = ("user-1",)) -> None:
        self._topics[chat_id] = topic
        self._users[chat_id] = frozenset(users)

    async def load(self, chat_id: str, apm_principal_token: str):
        if chat_id not in self._topics:
            raise ChatAccessError(f"chat {chat_id!r} not accessible")
        return SimpleNamespace(
            chat_id=chat_id, topic=self._topics[chat_id], user_ids=self._users[chat_id]
        )


class FakeAgentConfig:
    def __init__(self) -> None:
        self._snapshots: dict[str, AgentConfigSnapshot] = {}

    def register(self, snapshot: AgentConfigSnapshot) -> None:
        self._snapshots[snapshot.agent_id] = snapshot

    async def pin(self, tenant_id: str, agent_id: str, apm_principal_token: str):
        if agent_id not in self._snapshots:
            raise AgentConfigError(f"no configuration for agent={agent_id}")
        return self._snapshots[agent_id]


class FakePermission:
    def __init__(self) -> None:
        self._denied: set[str] = set()
        self.authorizations: list[str] = []

    def deny_voice(self, subject: str) -> None:
        self._denied.add(subject)

    async def has_voice_permission(
        self, principal: Principal, apm_principal_token: str, authorization: str
    ) -> bool:
        self.authorizations.append(authorization)
        return principal.subject not in self._denied


class FakeMcp:
    """Reports the tools/failures seeded per server, recording what discovery was asked."""

    def __init__(self) -> None:
        self.tools: dict[str, list[str]] = {}
        self.failures: dict[str, str] = {}
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.invocations: list[tuple[str, str, dict[str, str]]] = []

    async def discover(
        self, servers: list[McpServerSnapshot], headers: Mapping[str, str]
    ) -> McpDiscovery:
        self.calls.append(([s.name for s in servers], dict(headers)))
        discovery = McpDiscovery()
        for server in servers:
            if server.name in self.failures:
                discovery.degraded.append((server.name, self.failures[server.name]))
                continue
            for name in self.tools.get(server.name, []):
                discovery.tools.append(
                    AgentToolSnapshot(
                        tool_name=name, server_name=server.name, server_url=server.url
                    )
                )
        return discovery

    async def invoke(
        self,
        server: McpServerSnapshot,
        proposal: ToolProposal,
        headers: Mapping[str, str],
        args: dict[str, Any] | None = None,
    ) -> McpToolResult:
        self.invocations.append((server.name, proposal.tool_name, dict(headers)))
        return McpToolResult(ok=True, value="done")


def fake_container(
    *, token: str = "tok-user", chat_id: str = "chat-1", agent_id: str = "onecx-agent"
) -> AppContainer:
    """An app container with fakes for every OneCX boundary and one usable user/chat/agent.

    Turns are delimited by the client (push-to-talk) so tests drive them
    explicitly; VAD tests switch ``speech.turn_detection`` and use ``FakeVAD``.
    """
    container = AppContainer(
        Settings(
            llm=LlmSettings(provider_override="inmemory"),
            speech=SpeechSettings(turn_detection="client"),
        )
    )
    auth, chat, agent_config = FakeAuth(), FakeChat(), FakeAgentConfig()
    auth.seed(token, subject="user-1", tenant_id="local", roles=["voice"])
    chat.register(chat_id, topic="general")
    agent_config.register(
        AgentConfigSnapshot(agent_id=agent_id, schema_version="1", system_prompt="be brief")
    )
    container.auth = auth
    container.chat = chat
    container.agent_config = agent_config
    container.permission = FakePermission()
    container.mcp = FakeMcp()
    container.persistence = FakePersistence()  # type: ignore[assignment]
    return container


class FakePersistence:
    """A recording persistence that can be made to fail on the human/assistant write."""

    def __init__(
        self,
        *,
        fail_human: bool = False,
        fail_assistant: bool = False,
        human_delay: float = 0.0,
    ) -> None:
        self.human_delay = human_delay
        self.entries: list[VoiceEntry] = []
        self.fail_human = fail_human
        self.fail_assistant = fail_assistant
        self.appended_with: list[tuple[str, str, str]] = []
        self.history_messages: list[dict[str, str]] = []
        self.fail_history = False

    async def append(
        self,
        session_id: str,
        entry: VoiceEntry,
        *,
        apm_principal_token: str = "",
        user_id: str = "",
    ) -> str:
        self.appended_with.append((session_id, apm_principal_token, user_id))
        if entry.kind.value == "HUMAN" and self.human_delay:
            await asyncio.sleep(self.human_delay)
        if (entry.kind.value == "HUMAN" and self.fail_human) or (
            entry.kind.value == "ASSISTANT" and self.fail_assistant
        ):
            raise RuntimeError("persistence boom")
        entry_id = f"{session_id}:{len(self.entries) + 1}"
        self.entries.append(entry)
        return entry_id

    async def list_entries(self, session_id: str) -> list[VoiceEntry]:
        return list(self.entries)

    async def history(
        self, chat_id: str, *, apm_principal_token: str = "", limit: int = 50
    ) -> list[dict[str, str]]:
        if self.fail_history:
            raise PersistenceError("history boom")
        return self.history_messages[-limit:]


class FakeVAD(VADAnalyzer):
    """A deterministic VAD: any non-silent 10 ms frame is speech.

    Start/stop windows are short so tests can open a turn with a few non-zero
    chunks and close it with ~100 ms of zeros.
    """

    def __init__(self) -> None:
        super().__init__(params=VADParams(min_volume=0.0, start_secs=0.02, stop_secs=0.1))

    def num_frames_required(self) -> int:
        return 160

    def voice_confidence(self, buffer: bytes) -> float:
        return 1.0 if any(buffer) else 0.0
