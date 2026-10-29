import base64
import json

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TextFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

from onecx_pipecat.pipeline.serializer import BAD_COMMAND, OnecxFrameSerializer


@pytest.fixture
def serializer() -> OnecxFrameSerializer:
    return OnecxFrameSerializer(16000)


async def test_audio_becomes_an_input_audio_frame(serializer):
    frame = await serializer.deserialize(
        json.dumps({"type": "audio", "audio": base64.b64encode(b"\x01\x00\x02\x00").decode()})
    )
    assert isinstance(frame, InputAudioRawFrame)
    assert (frame.audio, frame.sample_rate, frame.num_channels) == (b"\x01\x00\x02\x00", 16000, 1)


async def test_turn_boundaries_become_proposed_user_turn_frames(serializer):
    started = await serializer.deserialize(json.dumps({"type": "user_started_speaking"}))
    stopped = await serializer.deserialize(json.dumps({"type": "user_stopped_speaking"}))
    assert isinstance(started, ProposedUserStartedSpeakingFrame)
    assert isinstance(stopped, ProposedUserStoppedSpeakingFrame)


@pytest.mark.parametrize(
    "payload", [{"type": "stop"}, {"type": "refresh_token", "apm_principal_token": "t"}]
)
async def test_control_commands_become_transport_messages(serializer, payload):
    frame = await serializer.deserialize(json.dumps(payload))
    assert isinstance(frame, InputTransportMessageFrame)
    assert frame.message == payload


@pytest.mark.parametrize(
    "data",
    [
        "not json",
        b"\xff\xfe",
        "[]",
        "42",
        json.dumps({"type": "nonsense"}),
        json.dumps({"no": "type"}),
        json.dumps({"type": "audio"}),
        json.dumps({"type": "audio", "audio": ""}),
        json.dumps({"type": "audio", "audio": "***"}),
        json.dumps({"type": "audio", "audio": 5}),
    ],
)
async def test_malformed_input_never_raises_and_is_flagged(serializer, data):
    frame = await serializer.deserialize(data)
    assert isinstance(frame, InputTransportMessageFrame)
    assert frame.message == BAD_COMMAND


async def _event(serializer, frame) -> dict | None:
    data = await serializer.serialize(frame)
    return None if data is None else json.loads(data)


async def test_transcripts_map_to_partial_and_final(serializer):
    partial = await _event(serializer, InterimTranscriptionFrame("hel", "u", "ts"))
    final = await _event(serializer, TranscriptionFrame("hello", "u", "ts"))
    assert partial == {"kind": "transcript", "text": "hel", "state": "PARTIAL"}
    assert final == {"kind": "transcript", "text": "hello", "state": "FINAL"}


async def test_spoken_text_maps_to_assistant_text(serializer):
    event = await _event(serializer, TTSTextFrame("hi there", aggregated_by="word"))
    assert event == {"kind": "assistant_text", "text": "hi there"}


async def test_output_audio_maps_to_base64_assistant_audio(serializer):
    frame = OutputAudioRawFrame(audio=b"\x01\x02", sample_rate=24000, num_channels=1)
    event = await _event(serializer, frame)
    assert event == {
        "kind": "assistant_audio",
        "audio": base64.b64encode(b"\x01\x02").decode(),
        "sample_rate": 24000,
        "num_channels": 1,
    }


async def test_bot_speaking_and_interruption_events(serializer):
    assert (await _event(serializer, BotStartedSpeakingFrame()))["kind"] == "bot_started_speaking"
    assert (await _event(serializer, BotStoppedSpeakingFrame()))["kind"] == "bot_stopped_speaking"
    assert (await _event(serializer, InterruptionFrame()))["kind"] == "interrupted"


async def test_user_turn_boundaries_are_reported(serializer):
    started = await _event(serializer, UserStartedSpeakingFrame())
    stopped = await _event(serializer, UserStoppedSpeakingFrame())
    assert started == {"kind": "user_started_speaking"}
    assert stopped == {"kind": "user_stopped_speaking"}


@pytest.mark.parametrize(
    "frame_type", [OutputTransportMessageFrame, OutputTransportMessageUrgentFrame]
)
async def test_transport_messages_are_sent_as_is(serializer, frame_type):
    message = {"kind": "turn_complete", "entry_id": "e1"}
    assert await _event(serializer, frame_type(message=message)) == message


async def test_other_frames_are_ignored(serializer):
    assert await serializer.serialize(TextFrame("x")) is None
    assert await serializer.serialize(OutputTransportMessageFrame(message="not a dict")) is None
