"""Small OneCX processors around the Pipecat pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pipecat.frames.frames import (
    CancelFrame,
    CancelWorkerFrame,
    Frame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputTransportMessageUrgentFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from onecx_pipecat.pipeline.captions import CaptionMessageFrame
from onecx_pipecat.pipeline.serializer import BAD_COMMAND, wire_event
from onecx_pipecat.protocol import (
    Command,
    ErrorEvent,
    Event,
    RefreshToken,
    StopSession,
    decode_command,
)

Emit = Callable[[Event], Awaitable[None]]


class SessionControl(FrameProcessor):
    """Handles ``refresh_token`` / ``stop`` and malformed commands from the client."""

    def __init__(
        self, dispatch: Callable[[Command], Awaitable[list[Event]]], emit: Emit, **kwargs
    ) -> None:
        super().__init__(name="SessionControl", **kwargs)
        self._dispatch = dispatch
        self._emit = emit

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputTransportMessageFrame):
            await self.push_frame(frame, direction)
            return
        message = frame.message if isinstance(frame.message, dict) else BAD_COMMAND
        command = decode_command(message)
        if not isinstance(command, RefreshToken | StopSession):
            await self._emit(ErrorEvent(code="bad_command", message="malformed command"))
        elif isinstance(command, StopSession):
            await self._dispatch(command)
            await self.push_frame(
                CancelWorkerFrame(reason="client_stopped"), FrameDirection.UPSTREAM
            )
        else:
            for event in await self._dispatch(command):
                await self._emit(event)


class WireEvents(FrameProcessor):
    """Mirrors client-visible frames as transport messages, then passes the frame on.

    The output transport only serializes interruptions, transport messages and
    audio, so frames like transcripts, captions and bot speaking state need an
    explicit transport message. It is urgent: sent immediately, not queued
    behind the audio that the output transport paces in real time.
    """

    def __init__(self, *, urgent: tuple[type[Frame], ...], **kwargs) -> None:
        super().__init__(name="WireEvents", **kwargs)
        self._urgent = urgent
        self._response_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseStartFrame):
            self._response_active = True
        elif direction is FrameDirection.DOWNSTREAM and isinstance(
            frame, InterruptionFrame | CancelFrame | CancelWorkerFrame
        ):
            self._response_active = False
        eligible = not isinstance(frame, TTSTextFrame) or self._response_active
        if (
            eligible
            and isinstance(frame, self._urgent)
            and (event := wire_event(frame)) is not None
        ):
            message = event.model_dump()
            if isinstance(frame, TTSTextFrame):
                await self.push_frame(
                    CaptionMessageFrame(message=message, context_id=frame.context_id)
                )
            else:
                await self.push_frame(OutputTransportMessageUrgentFrame(message=message))
        await self.push_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseEndFrame):
            self._response_active = False
