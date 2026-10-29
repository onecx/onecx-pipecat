"""Voice-session wire protocol (version 2).

The handshake commands (``start_session``) are plain JSON text messages handled
by the endpoint before the Pipecat pipeline starts. Once the session is READY
the socket belongs to the pipeline's ``OnecxFrameSerializer``,
which additionally understands ``audio`` (and, with client turn detection,
``user_started_speaking`` / ``user_stopped_speaking``) inbound and the
streaming events below outbound.

Commands are what a client asks a session to do; events are what the session
emits back. The vocabulary is independent of the carrier so the WebSocket can
later be replaced (for example with WebRTC) without changing session behavior.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

PROTOCOL_VERSION = 2


class StartSession(BaseModel):
    """Begin a voice session against a pinned agent configuration.

    Carries the user's OneCX tokens in the message body — never in the URL —
    so the session can authenticate them and load the chat as the user.
    ``apm_principal_token`` and ``authorization`` are the raw token values the
    user's browser presents (the latter is the ``Authorization`` header value).
    """

    type: Literal["start_session"] = "start_session"
    agent_id: str
    chat_id: str
    apm_principal_token: str
    authorization: str = ""


class RefreshToken(BaseModel):
    """Replace the session's OneCX tokens with freshly obtained values."""

    type: Literal["refresh_token"] = "refresh_token"
    apm_principal_token: str
    authorization: str = ""


class StopSession(BaseModel):
    """End the voice session and return to typed chat."""

    type: Literal["stop"] = "stop"


Command = StartSession | StopSession | RefreshToken


class SessionReady(BaseModel):
    kind: Literal["session_ready"] = "session_ready"
    session_id: str
    agent_id: str
    schema_version: str
    protocol_version: int = PROTOCOL_VERSION


class Transcript(BaseModel):
    kind: Literal["transcript"] = "transcript"
    text: str
    state: Literal["PARTIAL", "FINAL"]


class AssistantText(BaseModel):
    """A streamed chunk of the assistant's spoken text."""

    kind: Literal["assistant_text"] = "assistant_text"
    text: str


class AssistantAudio(BaseModel):
    kind: Literal["assistant_audio"] = "assistant_audio"
    audio: str
    sample_rate: int
    num_channels: int = 1


class BotStartedSpeaking(BaseModel):
    kind: Literal["bot_started_speaking"] = "bot_started_speaking"


class BotStoppedSpeaking(BaseModel):
    kind: Literal["bot_stopped_speaking"] = "bot_stopped_speaking"


class UserStartedSpeaking(BaseModel):
    """A user turn started; arriving while the bot speaks, it is a barge-in."""

    kind: Literal["user_started_speaking"] = "user_started_speaking"


class UserStoppedSpeaking(BaseModel):
    """The user turn ended; the transcript now goes to the LLM."""

    kind: Literal["user_stopped_speaking"] = "user_stopped_speaking"


class Interrupted(BaseModel):
    """The user barged in; the client must drop any queued assistant audio."""

    kind: Literal["interrupted"] = "interrupted"


class TurnComplete(BaseModel):
    kind: Literal["turn_complete"] = "turn_complete"
    entry_id: str


class DegradedCapability(BaseModel):
    """A capability is unavailable this session (e.g. an MCP server failed).

    The session continues with the remaining capabilities; this event lets a
    client surface the degradation to the user.
    """

    kind: Literal["degraded_capability"] = "degraded_capability"
    reason: str = ""


class ErrorEvent(BaseModel):
    kind: Literal["error"] = "error"
    code: str
    message: str


class SessionClosed(BaseModel):
    kind: Literal["session_closed"] = "session_closed"
    reason: str


Event = (
    SessionReady
    | Transcript
    | AssistantText
    | AssistantAudio
    | BotStartedSpeaking
    | BotStoppedSpeaking
    | UserStartedSpeaking
    | UserStoppedSpeaking
    | Interrupted
    | TurnComplete
    | ErrorEvent
    | DegradedCapability
    | SessionClosed
)


def _principal_token(raw: dict) -> str | None:
    """Read the user's ``apm-principal-token`` (or its underscore alias)."""
    for key in ("apm_principal_token", "apm-principal-token"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _optional_str(raw: dict, key: str) -> str:
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def decode_command(raw: dict) -> Command | None:
    """Turn a decoded JSON object into a typed command, or None if invalid."""
    match raw.get("type"):
        case "start_session":
            agent_id = raw.get("agent_id")
            chat_id = raw.get("chat_id")
            apm_principal_token = _principal_token(raw)
            if (
                not isinstance(agent_id, str)
                or not agent_id
                or not isinstance(chat_id, str)
                or not chat_id
                or apm_principal_token is None
            ):
                return None
            return StartSession(
                agent_id=agent_id,
                chat_id=chat_id,
                apm_principal_token=apm_principal_token,
                authorization=_optional_str(raw, "authorization"),
            )
        case "refresh_token":
            apm_principal_token = _principal_token(raw)
            if apm_principal_token is None:
                return None
            return RefreshToken(
                apm_principal_token=apm_principal_token,
                authorization=_optional_str(raw, "authorization"),
            )
        case "stop":
            return StopSession()
        case _:
            return None
