"""Tracks assistant captions emitted after synthesis but before audio finishes.

Captions (``assistant_text``) are sent after synthesis, while the assistant
aggregator after the output transport only collects the sentences that were
actually played. ``CaptionLedger`` keeps emitted sentences so an interrupted
reply is stored, and kept in the LLM context, as the text sent to the client.

The WebSocket output transport records a caption after a successful send.
``shown`` snapshots interruptions and ``settled`` tracks reply boundaries
after the output transport.
"""

from __future__ import annotations

import asyncio
from collections import deque

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
)

from onecx_pipecat.pipeline.serializer import caption_text
from onecx_pipecat.protocol import AssistantText

_CAPTION = AssistantText.model_fields["kind"].default
_REPLY_END = (LLMFullResponseEndFrame, LLMAssistantPushAggregationFrame)


class CaptionLedger:
    def __init__(self) -> None:
        # Captions sent to the client whose reply has not finished playing, oldest first.
        self._pending: deque[str] = deque()
        self._interrupted: deque[tuple[int | None, str]] = deque()
        self._reconciling: dict[int, asyncio.Event] = {}
        self.shown = _CaptionsShown(self)
        self.settled = _CaptionsSettled(self)

    def take_interrupted(self, frame_id: int | None = None) -> str:
        """Consume a snapshot, optionally matching the interruption that created it."""
        for index, (interruption_id, text) in enumerate(self._interrupted):
            if frame_id is None or interruption_id == frame_id:
                del self._interrupted[index]
                return text
        return ""

    def record(self, text: str) -> None:
        """A caption was successfully sent to the client transport."""
        self._pending.append(text)

    def interrupt(self, frame_id: int | None = None) -> None:
        """Snapshot the captions of every reply still playing."""
        text = " ".join(t.strip() for t in self._pending if t.strip())
        if text:
            self._interrupted.append((frame_id, text))
        self._pending.clear()

    def begin_reconciliation(self, frame_id: int) -> None:
        self._reconciling.setdefault(frame_id, asyncio.Event())

    def finish_reconciliation(self, frame_id: int) -> None:
        if event := self._reconciling.pop(frame_id, None):
            event.set()

    async def wait_for_reconciliation(self) -> None:
        while self._reconciling:
            await asyncio.gather(*(event.wait() for event in self._reconciling.values()))

    def settle(self, count: int) -> str:
        """Remove and return the captions of a reply that finished playing."""
        settled = []
        for _ in range(min(count, len(self._pending))):
            settled.append(self._pending.popleft().strip())
        return " ".join(text for text in settled if text)


class CaptionWebsocketOutputTransport(FastAPIWebsocketOutputTransport):
    """Records captions only after the WebSocket transport successfully sends them."""

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

    async def send_message(self, frame: OutputTransportMessageUrgentFrame) -> None:
        delivered = await self._write_frame(frame)
        message = frame.message
        if (
            delivered
            and isinstance(message, dict)
            and message.get("kind") == _CAPTION
            and isinstance(message.get("text"), str)
        ):
            self._ledger.record(message["text"])


class _CaptionsShown(FrameProcessor):
    def __init__(self, ledger: CaptionLedger) -> None:
        super().__init__(name="CaptionsShown")
        self._ledger = ledger

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame | CancelFrame):
            self._ledger.interrupt(frame.id)
        await self.push_frame(frame, direction)


class _CaptionsSettled(FrameProcessor):
    def __init__(self, ledger: CaptionLedger) -> None:
        super().__init__(name="CaptionsSettled")
        self._ledger = ledger
        self._emitted = 0
        self._boundaries: dict[int, int] = {}

    def complete(self, frame: Frame) -> str:
        """Settle a boundary when the assistant aggregator starts processing it."""
        return self._ledger.settle(self._boundaries.pop(frame.id, 0))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame | CancelFrame):
            self._emitted = 0
            self._boundaries.clear()
        elif direction is FrameDirection.DOWNSTREAM and caption_text(frame) is not None:
            self._emitted += 1
        elif direction is FrameDirection.DOWNSTREAM and isinstance(frame, _REPLY_END):
            self._boundaries[frame.id] = self._emitted
            self._emitted = 0
        await self.push_frame(frame, direction)
