"""Unit tests for the wire protocol: parse_inbound / serialize_outbound."""

import pytest
from pydantic import ValidationError

from onecx_pipecat.protocol import (
    AssistantAudioChunkEvent,
    AssistantTextEvent,
    AudioChunkCommand,
    CapabilityDegradedEvent,
    ErrorEvent,
    InterruptCommand,
    ListeningStartedEvent,
    ListeningStoppedEvent,
    ProcessingStartedEvent,
    SessionReadyEvent,
    SessionStoppedEvent,
    StartListeningCommand,
    StartSessionCommand,
    StopListeningCommand,
    StopSessionCommand,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    TurnInterruptedEvent,
    parse_inbound,
    serialize_outbound,
)

COMMAND_PAYLOADS = [
    {"type": "START_SESSION", "session_type": "AI_CHAT"},
    {"type": "STOP_SESSION"},
    {"type": "START_LISTENING"},
    {"type": "STOP_LISTENING"},
    {"type": "AUDIO_CHUNK", "audio_base64": "AAA=", "sequence": 1},
    {"type": "INTERRUPT"},
]


@pytest.mark.parametrize("raw", COMMAND_PAYLOADS)
def test_parse_inbound_accepts_every_command(raw):
    assert parse_inbound(raw).type.value == raw["type"]


def test_parse_inbound_rejects_unknown_type():
    with pytest.raises(ValidationError):
        parse_inbound({"type": "NOT_A_TYPE"})


EVENTS = [
    SessionReadyEvent(session_id="s1"),
    ListeningStartedEvent(),
    ListeningStoppedEvent(),
    TranscriptPartialEvent(text="turn on the", sequence=1),
    TranscriptFinalEvent(text="turn on the lights", sequence=2),
    ProcessingStartedEvent(),
    AssistantTextEvent(text="Sure,", sequence=1),
    AssistantAudioChunkEvent(audio_base64="AAE=", sequence=1),
    TurnInterruptedEvent(),
    SessionStoppedEvent(),
    CapabilityDegradedEvent(capability="assistant", reason="capability unavailable"),
    ErrorEvent(code="ADAPTER_FAILURE", message="assistant failed", fatal=True),
]


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: e.type.value)
def test_serialize_outbound_preserves_type(event):
    assert serialize_outbound(event)["type"] == event.type.value
