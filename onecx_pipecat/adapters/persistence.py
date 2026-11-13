"""Conversation persistence: save voice entries to a chat and load its history.

Persistence is keyed by chat id: a voice session is anchored to an existing
onecx-chat chat and shares its message history with the typed chat.
``HttpChatPersistence`` writes/reads through onecx-chat-svc; the in-memory
backend serves tests and local runs.
"""

from __future__ import annotations

from datetime import UTC as _UTC
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from onecx_pipecat.config import ChatSettings, PersistenceSettings
from onecx_pipecat.models import VoiceEntry, VoiceEntryKind

# An LLM context message: ``{"role": "user" | "assistant", "content": ...}``.
HistoryMessage = dict[str, str]

_ROLE_BY_MESSAGE_TYPE = {"HUMAN": "user", "ASSISTANT": "assistant"}


class PersistenceError(Exception):
    """Raised when entries cannot be saved or the history cannot be loaded."""


class ConversationPersistence(Protocol):
    async def append(
        self, chat_id: str, entry: VoiceEntry, *, apm_principal_token: str = "", user_id: str = ""
    ) -> str: ...

    async def history(
        self, chat_id: str, *, apm_principal_token: str = "", limit: int = 50
    ) -> list[HistoryMessage]: ...


class HttpChatPersistence:
    """Save and load messages through onecx-chat-svc as the user.

    Saves ``POST /internal/chats/{id}/messages`` (HUMAN -> ``HUMAN`` message
    with the user's id, ASSISTANT -> ``ASSISTANT`` message) with
    ``skipAIProcessing`` so chat-svc stores the message without forwarding it
    to the AI service. ``Message`` has no status field, so an INTERRUPTED
    assistant entry is stored as caption text sent to the client, spoken or not. Only
    visible HUMAN/ASSISTANT entries are written; other kinds are rejected.
    Loads ``GET /internal/chats/{id}/messages`` (404 means no messages yet).
    Failures raise ``PersistenceError`` so callers fail closed.
    """

    def __init__(
        self, settings: ChatSettings | None = None, service_auth: httpx.Auth | None = None
    ) -> None:
        self._settings = settings or ChatSettings()
        self._service_auth = service_auth
        self._client: httpx.AsyncClient | None = None

    def set_client(self, client: httpx.AsyncClient) -> None:
        """Inject a client (used by tests to avoid a live network call)."""
        self._client = client

    def _url(self, chat_id: str) -> str:
        return f"{self._settings.base_url}/internal/chats/{quote(chat_id, safe='')}/messages"

    async def _request(
        self, method: str, chat_id: str, apm_principal_token: str, **kwargs: Any
    ) -> httpx.Response:
        client = self._client or httpx.AsyncClient()
        try:
            return await client.request(
                method,
                self._url(chat_id),
                headers={"apm-principal-token": apm_principal_token},
                auth=self._service_auth,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise PersistenceError(f"chat service unreachable: {exc}") from exc
        finally:
            if client is not self._client:
                await client.aclose()

    async def append(
        self, chat_id: str, entry: VoiceEntry, *, apm_principal_token: str = "", user_id: str = ""
    ) -> str:
        if entry.kind is VoiceEntryKind.HUMAN:
            body: dict[str, Any] = {"type": "HUMAN", "text": entry.text, "userId": user_id}
        elif entry.kind is VoiceEntryKind.ASSISTANT:
            body = {"type": "ASSISTANT", "text": entry.text}
        else:
            raise PersistenceError(f"entry kind {entry.kind.value} is not persisted to the chat")
        body["skipAIProcessing"] = True
        body["agentConfigVersion"] = entry.agent_config_version
        response = await self._request("POST", chat_id, apm_principal_token, json=body)
        if not response.is_success:
            raise PersistenceError(f"chat service error: {response.status_code}")
        return _created_id(response)

    async def history(
        self, chat_id: str, *, apm_principal_token: str = "", limit: int = 50
    ) -> list[HistoryMessage]:
        response = await self._request("GET", chat_id, apm_principal_token)
        if response.status_code == 404:
            return []
        if not response.is_success:
            raise PersistenceError(f"chat service error: {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise PersistenceError("chat service returned a malformed response") from exc
        if not isinstance(body, list):
            raise PersistenceError("chat service returned a malformed response")
        return _to_history(body, limit)


def _created_id(response: httpx.Response) -> str:
    """The new message id: from the body if present, else the ``Location`` header."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("id"):
        return str(body["id"])
    location = response.headers.get("location", "").rstrip("/")
    if location:
        return location.rsplit("/", 1)[-1]
    raise PersistenceError("chat service did not return the created message id")


def _to_history(messages: list[Any], limit: int) -> list[HistoryMessage]:
    """Chronological user/assistant messages, newest ``limit`` kept.

    System messages, unknown types and blank texts are skipped. Messages are
    ordered by ``creationDate`` (ISO-8601); ties and missing dates keep the
    service's order.
    """
    dated: list[tuple[datetime, int, HistoryMessage]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = _ROLE_BY_MESSAGE_TYPE.get(str(message.get("type")))
        text = str(message.get("text") or "").strip()
        if role is None or not text:
            continue
        dated.append(
            (_parse_date(message.get("creationDate")), index, {"role": role, "content": text})
        )
    dated.sort(key=lambda item: (item[0], item[1]))
    kept = [message for _, _, message in dated]
    return kept[-limit:] if limit > 0 else []


def _parse_date(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.min.replace(tzinfo=_UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=_UTC)


class InMemoryPersistence:
    """Process-local storage; sufficient for the foundation and CI.

    Enforces ``persistence.max_entries_per_session``: when a session exceeds
    the cap the oldest entry is evicted so the per-session store stays within
    the configured soft limit.
    """

    def __init__(self, settings: PersistenceSettings | None = None) -> None:
        self._settings = settings or PersistenceSettings()
        self._sessions: dict[str, dict[str, VoiceEntry]] = {}
        self._sequence: dict[str, int] = {}

    async def append(
        self,
        session_id: str,
        entry: VoiceEntry,
        *,
        apm_principal_token: str = "",
        user_id: str = "",
    ) -> str:
        store = self._sessions.setdefault(session_id, {})
        self._sequence[session_id] = self._sequence.get(session_id, 0) + 1
        entry_id = f"{session_id}:{self._sequence[session_id]}"
        store[entry_id] = entry
        # Keep the store within the configured cap by evicting the oldest
        # entry when it is exceeded. The store only ever gains new unique
        # keys, so its insertion order is chronological and the first key is
        # the oldest entry.
        while len(store) > self._settings.max_entries_per_session:
            del store[next(iter(store))]
        return entry_id

    async def list_entries(self, session_id: str) -> list[VoiceEntry]:
        return list(self._sessions.get(session_id, {}).values())

    async def get(self, session_id: str, entry_id: str) -> VoiceEntry | None:
        return self._sessions.get(session_id, {}).get(entry_id)

    async def history(
        self, session_id: str, *, apm_principal_token: str = "", limit: int = 50
    ) -> list[HistoryMessage]:
        roles = {VoiceEntryKind.HUMAN: "user", VoiceEntryKind.ASSISTANT: "assistant"}
        messages = [
            {"role": roles[e.kind], "content": e.text}
            for e in self._sessions.get(session_id, {}).values()
            if e.kind in roles and not e.hidden and e.text.strip()
        ]
        return messages[-limit:] if limit > 0 else []
