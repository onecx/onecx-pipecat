"""Assembles the long-lived Pipecat pipeline of one voice session.

``transport.input() -> control -> stt -> transcripts -> user aggregator -> llm
-> tts -> wire events -> transport.output() -> assistant aggregator``

Turn boundaries come either from a VAD analyzer in the user aggregator
(``speech.turn_detection=vad``: the user turn starts on speech and ends after a
pause once a transcript arrived) or from the client (``user_started_speaking`` /
``user_stopped_speaking``) through Pipecat's external turn strategies. Either
way the user aggregator, interruption handling and the assistant aggregator are
stock Pipecat; only the OneCX pieces (control, wire events, persistence) are
custom.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import WebSocket
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies

from onecx_pipecat.config import Settings
from onecx_pipecat.models import VoiceEntry
from onecx_pipecat.pipeline.persistence import attach_persistence
from onecx_pipecat.pipeline.processors import SessionControl, WireEvents
from onecx_pipecat.pipeline.serializer import OnecxFrameSerializer
from onecx_pipecat.protocol import Command, ErrorEvent, Event

_ERROR_MESSAGES = {
    "stt_failed": "speech recognition failed",
    "llm_failed": "language model failed",
    "tts_failed": "speech synthesis failed",
}


def build_voice_pipeline(
    *,
    websocket: WebSocket,
    session_id: str,
    settings: Settings,
    system_prompt: str,
    tools: ToolsSchema,
    llm: FrameProcessor,
    stt: FrameProcessor,
    tts: FrameProcessor,
    history: list[dict[str, str]],
    save_entry: Callable[[VoiceEntry], Awaitable[str]],
    dispatch: Callable[[Command], Awaitable[list[Event]]],
    vad_analyzer: VADAnalyzer | None = None,
) -> PipelineWorker:
    sample_rate = settings.speech.sample_rate_hz
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=sample_rate,
            audio_out_sample_rate=sample_rate,
            audio_out_auto_silence=False,
            audio_out_end_silence_secs=0,
            serializer=OnecxFrameSerializer(sample_rate),
        ),
    )

    async def emit(event: Event) -> None:
        await worker.queue_frame(OutputTransportMessageUrgentFrame(message=event.model_dump()))

    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}, *history],
        **({"tools": tools} if tools.standard_tools else {}),
    )
    aggregators = LLMContextAggregatorPair(context, user_params=_user_params(vad_analyzer))
    user, assistant = aggregators.user(), aggregators.assistant()
    attach_persistence(user, assistant, save_entry, emit)

    pipeline = Pipeline(
        [
            transport.input(),
            SessionControl(dispatch, emit),
            stt,
            WireEvents(urgent=(InterimTranscriptionFrame, TranscriptionFrame)),
            user,
            llm,
            tts,
            WireEvents(
                urgent=(
                    BotStartedSpeakingFrame,
                    BotStoppedSpeakingFrame,
                    UserStartedSpeakingFrame,
                    UserStoppedSpeakingFrame,
                ),
                ordered=(TTSTextFrame,),
            ),
            transport.output(),
            assistant,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=sample_rate, audio_out_sample_rate=sample_rate),
        enable_rtvi=False,
        idle_timeout_secs=settings.pipeline.idle_timeout_secs,
    )

    sources = {id(stt): "stt_failed", id(llm): "llm_failed", id(tts): "tts_failed"}

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(_worker, frame) -> None:
        if code := sources.get(id(frame.processor)):
            await emit(ErrorEvent(code=code, message=_ERROR_MESSAGES[code]))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _websocket) -> None:
        await worker.cancel(reason="client_disconnected")

    return worker


def _user_params(vad_analyzer: VADAnalyzer | None) -> LLMUserAggregatorParams:
    if vad_analyzer is None:
        return LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies())
    # Default start strategies (VAD, transcription); the turn stops after a
    # pause once a transcript arrived, without loading a turn-detection model.
    return LLMUserAggregatorParams(
        vad_analyzer=vad_analyzer,
        user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()]),
    )
