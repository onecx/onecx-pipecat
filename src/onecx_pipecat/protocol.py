"""Wire protocol: one Pydantic model per inbound command and outbound event.

``InboundCommand`` and ``OutboundEvent`` are discriminated unions keyed on
the ``type`` field. ``parse_inbound`` and ``serialize_outbound`` are the only
serialization entry points the transport uses.
"""

from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class CommandType(StrEnum):
    START_SESSION = "START_SESSION"
    STOP_SESSION = "STOP_SESSION"
    START_LISTENING = "START_LISTENING"
    STOP_LISTENING = "STOP_LISTENING"
    AUDIO_CHUNK = "AUDIO_CHUNK"
    INTERRUPT = "INTERRUPT"


class EventType(StrEnum):
    SESSION_READY = "SESSION_READY"
    LISTENING_STARTED = "LISTENING_STARTED"
    LISTENING_STOPPED = "LISTENING_STOPPED"
    TRANSCRIPT_PARTIAL = "TRANSCRIPT_PARTIAL"
    TRANSCRIPT_FINAL = "TRANSCRIPT_FINAL"
    PROCESSING_STARTED = "PROCESSING_STARTED"
    ASSISTANT_TEXT = "ASSISTANT_TEXT"
    ASSISTANT_AUDIO_CHUNK = "ASSISTANT_AUDIO_CHUNK"
    TURN_INTERRUPTED = "TURN_INTERRUPTED"
    SESSION_STOPPED = "SESSION_STOPPED"
    CAPABILITY_DEGRADED = "CAPABILITY_DEGRADED"
    ERROR = "ERROR"


class StartSessionCommand(BaseModel):
    type: Literal[CommandType.START_SESSION] = CommandType.START_SESSION
    session_type: Literal["AI_CHAT"]


class StopSessionCommand(BaseModel):
    type: Literal[CommandType.STOP_SESSION] = CommandType.STOP_SESSION


class StartListeningCommand(BaseModel):
    type: Literal[CommandType.START_LISTENING] = CommandType.START_LISTENING


class StopListeningCommand(BaseModel):
    type: Literal[CommandType.STOP_LISTENING] = CommandType.STOP_LISTENING


class AudioChunkCommand(BaseModel):
    type: Literal[CommandType.AUDIO_CHUNK] = CommandType.AUDIO_CHUNK
    audio_base64: str
    sequence: int


class InterruptCommand(BaseModel):
    type: Literal[CommandType.INTERRUPT] = CommandType.INTERRUPT


class SessionReadyEvent(BaseModel):
    type: Literal[EventType.SESSION_READY] = EventType.SESSION_READY
    session_id: str


class ListeningStartedEvent(BaseModel):
    type: Literal[EventType.LISTENING_STARTED] = EventType.LISTENING_STARTED


class ListeningStoppedEvent(BaseModel):
    type: Literal[EventType.LISTENING_STOPPED] = EventType.LISTENING_STOPPED


class TranscriptPartialEvent(BaseModel):
    type: Literal[EventType.TRANSCRIPT_PARTIAL] = EventType.TRANSCRIPT_PARTIAL
    text: str
    sequence: int


class TranscriptFinalEvent(BaseModel):
    type: Literal[EventType.TRANSCRIPT_FINAL] = EventType.TRANSCRIPT_FINAL
    text: str
    sequence: int


class ProcessingStartedEvent(BaseModel):
    type: Literal[EventType.PROCESSING_STARTED] = EventType.PROCESSING_STARTED


class AssistantTextEvent(BaseModel):
    type: Literal[EventType.ASSISTANT_TEXT] = EventType.ASSISTANT_TEXT
    text: str
    sequence: int


class AssistantAudioChunkEvent(BaseModel):
    type: Literal[EventType.ASSISTANT_AUDIO_CHUNK] = EventType.ASSISTANT_AUDIO_CHUNK
    audio_base64: str
    sequence: int


class TurnInterruptedEvent(BaseModel):
    type: Literal[EventType.TURN_INTERRUPTED] = EventType.TURN_INTERRUPTED


class SessionStoppedEvent(BaseModel):
    type: Literal[EventType.SESSION_STOPPED] = EventType.SESSION_STOPPED


class CapabilityDegradedEvent(BaseModel):
    type: Literal[EventType.CAPABILITY_DEGRADED] = EventType.CAPABILITY_DEGRADED
    capability: str
    reason: str


class ErrorEvent(BaseModel):
    type: Literal[EventType.ERROR] = EventType.ERROR
    code: str
    message: str
    fatal: bool


InboundCommand = Annotated[
    Union[
        StartSessionCommand,
        StopSessionCommand,
        StartListeningCommand,
        StopListeningCommand,
        AudioChunkCommand,
        InterruptCommand,
    ],
    Field(discriminator="type"),
]
_inbound_adapter = TypeAdapter(InboundCommand)

OutboundEvent = Annotated[
    Union[
        SessionReadyEvent,
        ListeningStartedEvent,
        ListeningStoppedEvent,
        TranscriptPartialEvent,
        TranscriptFinalEvent,
        ProcessingStartedEvent,
        AssistantTextEvent,
        AssistantAudioChunkEvent,
        TurnInterruptedEvent,
        SessionStoppedEvent,
        CapabilityDegradedEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]
_outbound_adapter = TypeAdapter(OutboundEvent)


def parse_inbound(raw: dict) -> InboundCommand:
    """Validate a raw inbound JSON payload against the command union."""
    return _inbound_adapter.validate_python(raw)


def serialize_outbound(event: OutboundEvent) -> dict:
    """Serialize an outbound event into a JSON-ready dict."""
    return _outbound_adapter.dump_python(event, mode="json")
