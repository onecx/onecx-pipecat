"""Wire serializer for the voice WebSocket.

Inbound JSON becomes Pipecat frames; outbound frames become JSON events. The
frame -> event mapping lives in :func:`wire_event` so the same mapping is used
by the serializer and by :class:`~onecx_pipecat.pipeline.processors.WireEvents`,
which converts frames the output transport never serializes on its own
(transcripts, assistant text, bot speaking state) into transport messages.

``deserialize`` must never raise: an exception would end the transport's
receive loop. Malformed input is turned into a ``BAD_COMMAND`` control message
that the session control processor reports as ``error{bad_command}``.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

from onecx_pipecat.protocol import (
    AssistantAudio,
    AssistantText,
    BotStartedSpeaking,
    BotStoppedSpeaking,
    Event,
    Interrupted,
    Transcript,
    UserStartedSpeaking,
    UserStoppedSpeaking,
)

BAD_COMMAND = {"type": "bad_command"}


def wire_event(frame: Frame) -> Event | None:
    """The client event a frame maps to, or None when it is not client-visible."""
    if isinstance(frame, InterimTranscriptionFrame):
        return Transcript(text=frame.text, state="PARTIAL")
    if isinstance(frame, TranscriptionFrame):
        return Transcript(text=frame.text, state="FINAL")
    if isinstance(frame, TTSTextFrame):
        return AssistantText(text=frame.text)
    if isinstance(frame, OutputAudioRawFrame):
        return AssistantAudio(
            audio=base64.b64encode(frame.audio).decode("ascii"),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
        )
    if isinstance(frame, BotStartedSpeakingFrame):
        return BotStartedSpeaking()
    if isinstance(frame, BotStoppedSpeakingFrame):
        return BotStoppedSpeaking()
    if isinstance(frame, UserStartedSpeakingFrame):
        return UserStartedSpeaking()
    if isinstance(frame, UserStoppedSpeakingFrame):
        return UserStoppedSpeaking()
    if isinstance(frame, InterruptionFrame):
        return Interrupted()
    return None


class OnecxFrameSerializer(FrameSerializer):
    def __init__(self, sample_rate: int = 16000, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sample_rate = sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputTransportMessageFrame | OutputTransportMessageUrgentFrame):
            message = frame.message
            return json.dumps(message) if isinstance(message, dict) else None
        event = wire_event(frame)
        return event.model_dump_json() if event is not None else None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            payload = json.loads(data)
        except (ValueError, TypeError):
            return _bad_command()
        if not isinstance(payload, dict):
            return _bad_command()
        match payload.get("type"):
            case "audio":
                return self._audio(payload.get("audio"))
            case "user_started_speaking":
                return ProposedUserStartedSpeakingFrame()
            case "user_stopped_speaking":
                return ProposedUserStoppedSpeakingFrame()
            case "refresh_token" | "stop":
                return InputTransportMessageFrame(message=payload)
        return _bad_command()

    def _audio(self, encoded: object) -> Frame:
        if not isinstance(encoded, str) or not encoded:
            return _bad_command()
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return _bad_command()
        return InputAudioRawFrame(audio=audio, sample_rate=self._sample_rate, num_channels=1)


def _bad_command() -> InputTransportMessageFrame:
    return InputTransportMessageFrame(message=dict(BAD_COMMAND))
