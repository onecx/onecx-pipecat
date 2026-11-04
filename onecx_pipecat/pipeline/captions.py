"""Accounts for written assistant captions before audio playback finishes.

Captions are written urgently after synthesis, independently of audio playback.
The output transport records only successful writes and snapshots pending
captions before forwarding an interruption. TTS context IDs bind recorded
captions to response boundaries without changing the wire protocol.

Pipecat swallows WebSocket send errors; a successful write is not a browser
receipt or render acknowledgement.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMFullResponseEndFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketClient,
    FastAPIWebsocketOutputTransport,
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from onecx_pipecat.pipeline.serializer import caption_text
from onecx_pipecat.protocol import AssistantText

_CAPTION = AssistantText.model_fields["kind"].default
_REPLY_END = (LLMFullResponseEndFrame, LLMAssistantPushAggregationFrame)


@dataclass
class CaptionMessageFrame(OutputTransportMessageUrgentFrame):
    context_id: str | None


class CaptionLedger:
    def __init__(self) -> None:
        # Captions sent to the client whose reply has not finished playing, oldest first.
        self._pending: deque[tuple[str | None, str]] = deque()
        self._interrupted: deque[tuple[int | None, str]] = deque()
        self.settled = _CaptionsSettled(self)

    def take_interrupted(self, frame_id: int | None = None) -> str:
        """Consume a snapshot, optionally matching the interruption that created it."""
        for index, (interruption_id, text) in enumerate(self._interrupted):
            if frame_id is None or interruption_id == frame_id:
                del self._interrupted[index]
                return text
        return ""

    def record(self, context_id: str | None, text: str) -> None:
        """A caption was successfully sent to the client transport."""
        self._pending.append((context_id, text))

    def interrupt(self, frame_id: int | None = None) -> None:
        """Snapshot the captions of every reply still playing."""
        text = " ".join(text.strip() for _, text in self._pending if text.strip())
        if text:
            self._interrupted.append((frame_id, text))
        self._pending.clear()

    def settle(self, context_ids: list[str | None]) -> str:
        """Remove and return the captions of a reply that finished playing."""
        ids = set(context_ids)
        settled = []
        pending = deque()
        while self._pending:
            context_id, text = self._pending.popleft()
            if context_id in ids:
                settled.append(text.strip())
            else:
                pending.append((context_id, text))
        self._pending = pending
        return " ".join(text for text in settled if text)


class CaptionWebsocketOutputTransport(FastAPIWebsocketOutputTransport):
    """Records captions only when Pipecat reports a successful WebSocket write.

    Pipecat swallows socket send errors, so a successful write does not prove
    that the browser received or rendered the caption.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        *,
        ledger: CaptionLedger,
        **kwargs,
    ) -> None:
        super().__init__(transport, client, params, **kwargs)
        self._ledger = ledger

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if direction is FrameDirection.DOWNSTREAM and isinstance(
            frame, InterruptionFrame | CancelFrame
        ):
            self._ledger.interrupt(frame.id)
        await super().process_frame(frame, direction)

    async def send_message(self, frame: OutputTransportMessageUrgentFrame) -> None:
        delivered = await self._write_frame(frame)
        message = frame.message
        if (
            delivered
            and isinstance(frame, CaptionMessageFrame)
            and isinstance(message, dict)
            and message.get("kind") == _CAPTION
            and isinstance(message.get("text"), str)
        ):
            self._ledger.record(frame.context_id, message["text"])


class _CaptionsSettled(FrameProcessor):
    def __init__(self, ledger: CaptionLedger) -> None:
        super().__init__(name="CaptionsSettled")
        self._ledger = ledger
        self._context_ids: list[str | None] = []
        self._boundaries: dict[int, list[str | None]] = {}

    def complete(self, frame: Frame) -> str:
        """Settle a boundary when the assistant aggregator starts processing it."""
        return self._ledger.settle(self._boundaries.pop(frame.id, []))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame | CancelFrame):
            self._context_ids.clear()
            self._boundaries.clear()
        elif direction is FrameDirection.DOWNSTREAM and caption_text(frame) is not None:
            self._context_ids.append(getattr(frame, "context_id", None))
        elif direction is FrameDirection.DOWNSTREAM and isinstance(frame, _REPLY_END):
            self._boundaries[frame.id] = self._context_ids
            self._context_ids = []
        await self.push_frame(frame, direction)


def install_caption_transport(
    transport: FastAPIWebsocketTransport, ledger: CaptionLedger
) -> CaptionWebsocketOutputTransport:
    """Replace Pipecat's output processor using the pinned 1.8 transport API."""
    output = transport.output()
    if not isinstance(output, FastAPIWebsocketOutputTransport):
        raise RuntimeError("Pipecat websocket transport output has an unsupported type")
    replacement = CaptionWebsocketOutputTransport(
        transport,
        transport._client,
        transport._params,
        ledger=ledger,
        name=transport._output_name,
    )
    transport._output = replacement
    if transport.output() is not replacement:
        raise RuntimeError("Could not install caption websocket output transport")
    return replacement
