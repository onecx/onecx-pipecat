"""pytest-asyncio tests exercising VoiceSession directly against the fakes."""

import pytest

from onecx_pipecat.adapters import (
    DegradedFakeAssistantAdapter,
    FakeAssistantAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
)
from onecx_pipecat.context import SessionContext
from onecx_pipecat.protocol import (
    AudioChunkCommand,
    InterruptCommand,
    StartListeningCommand,
    StartSessionCommand,
    StopListeningCommand,
    StopSessionCommand,
)
from onecx_pipecat.session import VoiceSession

pytestmark = pytest.mark.asyncio


def make_context() -> SessionContext:
    return SessionContext(
        session_id="s1", tenant_id="tenant-a", principal_id="user-1", session_type="AI_CHAT"
    )


async def test_full_turn_emits_expected_event_sequence():
    stt, tts, assistant = FakeSpeechToTextAdapter(), FakeTextToSpeechAdapter(), FakeAssistantAdapter()
    session = VoiceSession(make_context(), stt, tts, assistant)
    events = []
    for command in (
        StartSessionCommand(session_type="AI_CHAT"),
        StartListeningCommand(),
        AudioChunkCommand(audio_base64="AAA=", sequence=1),
        StopListeningCommand(),
        StopSessionCommand(),
    ):
        async for event in session.handle_command(command):
            events.append(event.type.value)
    assert events == [
        "SESSION_READY",
        "LISTENING_STARTED",
        "LISTENING_STOPPED",
        "PROCESSING_STARTED",
        "TRANSCRIPT_PARTIAL",
        "TRANSCRIPT_FINAL",
        "ASSISTANT_TEXT",
        "ASSISTANT_TEXT",
        "ASSISTANT_TEXT",
        "ASSISTANT_AUDIO_CHUNK",
        "ASSISTANT_AUDIO_CHUNK",
        "SESSION_STOPPED",
    ]


async def test_audio_chunk_rejected_before_listening():
    stt, tts, assistant = FakeSpeechToTextAdapter(), FakeTextToSpeechAdapter(), FakeAssistantAdapter()
    session = VoiceSession(make_context(), stt, tts, assistant)
    async for _ in session.handle_command(StartSessionCommand(session_type="AI_CHAT")):
        pass
    events = [event async for event in session.handle_command(AudioChunkCommand(audio_base64="AAA=", sequence=1))]
    assert len(events) == 1
    assert events[0].type.value == "ERROR"
    assert events[0].code == "ILLEGAL_MICROPHONE_TRANSITION"
    assert session.session_fsm.state.value == "READY"


async def test_interrupt_during_processing_stops_turn():
    stt, tts, assistant = FakeSpeechToTextAdapter(), FakeTextToSpeechAdapter(), FakeAssistantAdapter()
    session = VoiceSession(make_context(), stt, tts, assistant)
    for command in (
        StartSessionCommand(session_type="AI_CHAT"),
        StartListeningCommand(),
        AudioChunkCommand(audio_base64="AAA=", sequence=1),
    ):
        async for _ in session.handle_command(command):
            pass
    session.session_fsm.apply("begin_processing")
    events = [event async for event in session.handle_command(InterruptCommand())]
    assert len(events) == 1
    assert events[0].type.value == "TURN_INTERRUPTED"
    assert session.session_fsm.state.value == "READY"


async def test_degraded_assistant_emits_capability_degraded_and_error():
    session = VoiceSession(
        make_context(),
        FakeSpeechToTextAdapter(),
        FakeTextToSpeechAdapter(),
        DegradedFakeAssistantAdapter(),
    )
    for command in (
        StartSessionCommand(session_type="AI_CHAT"),
        StartListeningCommand(),
        AudioChunkCommand(audio_base64="AAA=", sequence=1),
    ):
        async for _ in session.handle_command(command):
            pass
    events = [event async for event in session.handle_command(StopListeningCommand())]
    assert [e.type.value for e in events][-2:] == ["CAPABILITY_DEGRADED", "ERROR"]
    assert events[-1].fatal is True


async def test_close_releases_adapters_and_is_idempotent():
    stt, tts, assistant = FakeSpeechToTextAdapter(), FakeTextToSpeechAdapter(), FakeAssistantAdapter()
    session = VoiceSession(make_context(), stt, tts, assistant)
    await session.close()
    assert stt.closed is True
    assert tts.closed is True
    assert assistant.closed is True
    await session.close()


async def test_tenant_and_principal_propagated_to_adapters():
    stt, tts, assistant = FakeSpeechToTextAdapter(), FakeTextToSpeechAdapter(), FakeAssistantAdapter()
    context = make_context()
    session = VoiceSession(context, stt, tts, assistant)
    for command in (
        StartSessionCommand(session_type="AI_CHAT"),
        StartListeningCommand(),
        AudioChunkCommand(audio_base64="AAA=", sequence=1),
        StopListeningCommand(),
    ):
        async for _ in session.handle_command(command):
            pass
    assert stt.received_contexts == [context]
    assert assistant.received_contexts == [context]
    assert tts.received_contexts == [context]
