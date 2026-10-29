"""Deterministic in-memory STT, LLM and TTS processors for tests and local development.

They speak the same frames as the real Pipecat services so the pipeline above
them (turn strategies, aggregators, interruption handling) runs unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601

IN_MEMORY_TRANSCRIPT = "in-memory transcript"

# Speech boundaries from the client (push-to-talk) or from the VAD analyzer.
# A VAD start is not a reset: the VAD reports speech only after hearing some of
# it, and that audio belongs to the utterance.
_SPEECH_STARTED = (ProposedUserStartedSpeakingFrame,)
_SPEECH_STOPPED = (ProposedUserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)


class InMemorySTT(FrameProcessor):
    """Emits a fixed transcript when speech ends after audio was received."""

    def __init__(self, transcript: str = IN_MEMORY_TRANSCRIPT, **kwargs: Any) -> None:
        super().__init__(name="InMemorySTT", **kwargs)
        self._transcript = transcript
        self._heard_audio = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and frame.audio:
            self._heard_audio = True
        elif isinstance(frame, _SPEECH_STARTED):
            self._heard_audio = False
        elif isinstance(frame, _SPEECH_STOPPED) and self._heard_audio:
            self._heard_audio = False
            await self.push_frame(
                InterimTranscriptionFrame(self._transcript, "user", time_now_iso8601())
            )
            await self.push_frame(TranscriptionFrame(self._transcript, "user", time_now_iso8601()))
        await self.push_frame(frame, direction)


class InMemoryLLM(FrameProcessor):
    """Answers every context with a fixed or echoing reply; never calls a provider."""

    def __init__(self, reply: str | None = None, **kwargs: Any) -> None:
        super().__init__(name="InMemoryLLM", **kwargs)
        self._reply = reply
        self.functions: dict[str, Any] = {}

    def register_function(self, function_name: str | None, handler: Any, **_: Any) -> None:
        self.functions[function_name or ""] = handler

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(self._reply or _echo(frame)))
        await self.push_frame(LLMFullResponseEndFrame())


def _echo(frame: LLMContextFrame) -> str:
    for message in reversed(frame.context.messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return f"in-memory reply to: {message.get('content')}"
    return "in-memory reply"


class InMemoryTTS(TTSService):
    """A real ``TTSService`` that synthesizes silent PCM chunks of a fixed length."""

    def __init__(
        self, *, sample_rate: int = 16000, chunks: int = 2, chunk_ms: int = 20, **kwargs: Any
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )
        self._chunks = chunks
        self._chunk_bytes = sample_rate * chunk_ms // 1000 * 2

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        for _ in range(self._chunks):
            yield TTSAudioRawFrame(
                b"\x00" * self._chunk_bytes, self.sample_rate, 1, context_id=context_id
            )
