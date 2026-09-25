"""``VoiceSession``: the orchestrator binding context, state machines, and adapters.

``handle_command`` dispatches each inbound command through the state machines
and adapters and yields outbound events. ``close`` is idempotent and releases
every adapter exactly once.
"""

import base64
from typing import AsyncIterator

from .adapters.base import AssistantAdapter, SpeechToTextAdapter, TextToSpeechAdapter
from .context import SessionContext
from .errors import (
    AdapterFailureError,
    IllegalMicrophoneTransitionError,
    IllegalSessionTransitionError,
)
from .protocol import (
    AssistantAudioChunkEvent,
    AssistantTextEvent,
    CapabilityDegradedEvent,
    CommandType,
    ErrorEvent,
    InboundCommand,
    ListeningStartedEvent,
    ListeningStoppedEvent,
    OutboundEvent,
    ProcessingStartedEvent,
    SessionReadyEvent,
    SessionStoppedEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    TurnInterruptedEvent,
)
from .session_state import MicrophoneStateMachine, SessionStateMachine


class VoiceSession:
    """Owns one session and one microphone state machine, three adapters, and a context."""

    def __init__(
        self,
        context: SessionContext,
        stt: SpeechToTextAdapter,
        tts: TextToSpeechAdapter,
        assistant: AssistantAdapter,
    ) -> None:
        self.context = context
        self.stt = stt
        self.tts = tts
        self.assistant = assistant
        self.session_fsm = SessionStateMachine()
        self.mic_fsm = MicrophoneStateMachine()
        self._pending_audio: list[bytes] = []
        self._closed = False

    async def handle_command(self, command: InboundCommand) -> AsyncIterator[OutboundEvent]:
        try:
            match command.type:
                case CommandType.START_SESSION:
                    self.session_fsm.apply("start")
                    yield SessionReadyEvent(session_id=self.context.session_id)
                case CommandType.START_LISTENING:
                    self.mic_fsm.apply("enable")
                    self.session_fsm.apply("begin_listening")
                    self._pending_audio = []
                    yield ListeningStartedEvent()
                case CommandType.AUDIO_CHUNK:
                    if self.mic_fsm.state.value != "LISTENING":
                        raise IllegalMicrophoneTransitionError(
                            self.mic_fsm.state.value, "receive_audio"
                        )
                    elif self.session_fsm.state.value != "LISTENING":
                        raise IllegalSessionTransitionError(
                            self.session_fsm.state.value, "receive_audio"
                        )
                    else:
                        self._pending_audio.append(base64.b64decode(command.audio_base64))
                case CommandType.STOP_LISTENING:
                    self.mic_fsm.apply("disable")
                    self.session_fsm.apply("begin_processing")
                    yield ListeningStoppedEvent()
                    yield ProcessingStartedEvent()
                    async for event in self._run_turn():
                        yield event
                case CommandType.INTERRUPT:
                    if self.session_fsm.state.value not in ("PROCESSING", "RESPONDING"):
                        raise IllegalSessionTransitionError(
                            self.session_fsm.state.value, "interrupt"
                        )
                    else:
                        self.session_fsm.apply("finish_turn")
                        yield TurnInterruptedEvent()
                case CommandType.STOP_SESSION:
                    self.session_fsm.apply("stop")
                    await self.close()
                    yield SessionStoppedEvent()
        except (IllegalSessionTransitionError, IllegalMicrophoneTransitionError) as exc:
            yield ErrorEvent(code=exc.error_code, message=exc.message, fatal=False)
        except AdapterFailureError as exc:
            yield CapabilityDegradedEvent(capability=exc.adapter_name, reason=exc.message)
            self.session_fsm.apply("fault")
            yield ErrorEvent(code=exc.error_code, message=exc.message, fatal=True)

    async def _run_turn(self) -> AsyncIterator[OutboundEvent]:
        async def _audio_iter():
            for chunk in self._pending_audio:
                yield chunk

        final_text = ""
        async for chunk in self.stt.stream_transcripts(self.context, _audio_iter()):
            if chunk.is_final:
                final_text = chunk.text
                yield TranscriptFinalEvent(text=chunk.text, sequence=chunk.sequence)
            else:
                yield TranscriptPartialEvent(text=chunk.text, sequence=chunk.sequence)

        self.session_fsm.apply("begin_responding")
        sequence = 1
        async for text_chunk in self.assistant.respond(self.context, final_text):
            yield AssistantTextEvent(text=text_chunk, sequence=sequence)
            sequence += 1
        sequence = 1
        async for audio_chunk in self.tts.synthesize(self.context, final_text):
            yield AssistantAudioChunkEvent(
                audio_base64=base64.b64encode(audio_chunk).decode("ascii"),
                sequence=sequence,
            )
            sequence += 1
        self.session_fsm.apply("finish_turn")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.stt.aclose()
        await self.tts.aclose()
        await self.assistant.aclose()

    async def __aenter__(self) -> "VoiceSession":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()
