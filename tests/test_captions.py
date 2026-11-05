from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputTransportMessageUrgentFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketOutputTransport,
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from starlette.websockets import WebSocketState

from onecx_pipecat.pipeline.captions import (
    CaptionLedger,
    CaptionMessageFrame,
    CaptionWebsocketOutputTransport,
    install_caption_transport,
)
from onecx_pipecat.pipeline.processors import WireEvents
from onecx_pipecat.pipeline.serializer import OnecxFrameSerializer


class _Output(CaptionWebsocketOutputTransport):
    def __init__(self, ledger: CaptionLedger, *, delivered: bool = True) -> None:
        super().__init__(
            SimpleNamespace(),
            SimpleNamespace(),
            FastAPIWebsocketParams(serializer=OnecxFrameSerializer()),
            ledger=ledger,
        )
        self.delivered = delivered

    async def _write_frame(self, _frame) -> bool:
        return self.delivered

    async def push_frame(self, _frame, _direction) -> None:
        pass

    async def _handle_frame(self, _frame) -> None:
        pass


def _tts(text: str, context_id: str) -> TTSTextFrame:
    frame = TTSTextFrame(text, aggregated_by="sentence", context_id=context_id)
    frame.will_be_spoken = True
    return frame


async def _wire_caption(frame: TTSTextFrame) -> CaptionMessageFrame:
    wire = WireEvents(urgent=(TTSTextFrame,))
    outgoing = []

    async def capture(message, _direction=None) -> None:
        outgoing.append(message)

    wire.push_frame = capture
    await wire.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await wire.process_frame(frame, FrameDirection.DOWNSTREAM)
    return next(message for message in outgoing if isinstance(message, CaptionMessageFrame))


async def _settle(ledger: CaptionLedger, context_id: str) -> str:
    settled = ledger.settled

    async def discard(_frame, _direction) -> None:
        pass

    settled.push_frame = discard
    caption = _tts("boundary", context_id)
    await settled.process_frame(caption, FrameDirection.DOWNSTREAM)
    boundary = LLMFullResponseEndFrame()
    await settled.process_frame(boundary, FrameDirection.DOWNSTREAM)
    return settled.complete(boundary)


async def test_transport_snapshots_every_caption_written_before_interruption(monkeypatch):
    async def skip_base_process_frame(_self, _frame, _direction):
        pass

    monkeypatch.setattr(FastAPIWebsocketOutputTransport, "process_frame", skip_base_process_frame)
    ledger = CaptionLedger()
    output = _Output(ledger)
    for text in ("Sentence 1.", "Sentence 2."):
        await output.send_message(await _wire_caption(_tts(text, "reply-1")))
    await output.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    assert ledger.take_interrupted() == "Sentence 1. Sentence 2."


async def test_failed_write_is_not_recorded_or_consumed_by_later_reply():
    ledger = CaptionLedger()
    output = _Output(ledger, delivered=False)
    await output.send_message(await _wire_caption(_tts("Unwritten.", "reply-1")))
    assert await _settle(ledger, "reply-1") == ""
    output.delivered = True
    await output.send_message(await _wire_caption(_tts("Next reply.", "reply-2")))
    assert await _settle(ledger, "reply-2") == "Next reply."


async def test_non_caption_transport_messages_are_not_recorded():
    ledger = CaptionLedger()
    output = _Output(ledger)
    await output.send_message(
        OutputTransportMessageUrgentFrame(message={"kind": "transcript", "text": "user words"})
    )
    assert ledger.settle(["reply-1"]) == ""


async def test_settling_uses_context_identity_instead_of_caption_count():
    ledger = CaptionLedger()
    ledger.record("reply-1", "First reply.")
    ledger.record("reply-2", "Second reply.")
    assert ledger.settle(["reply-2"]) == "Second reply."
    ledger.interrupt()
    assert ledger.take_interrupted() == "First reply."


def test_interrupted_snapshots_are_matched_to_their_interruption():
    ledger = CaptionLedger()
    ledger.record("first", "First reply.")
    ledger.interrupt(1)
    ledger.record("second", "Second reply.")
    ledger.interrupt(2)
    assert ledger.take_interrupted(2) == "Second reply."
    assert ledger.take_interrupted(1) == "First reply."


@pytest.mark.parametrize("closing", [False, True])
async def test_real_transport_write_returns_false_for_a_closed_client(closing):
    websocket = SimpleNamespace(
        headers={},
        client_state=(WebSocketState.CONNECTED if closing else WebSocketState.DISCONNECTED),
        application_state=WebSocketState.CONNECTED,
    )
    transport = FastAPIWebsocketTransport(
        websocket, FastAPIWebsocketParams(serializer=OnecxFrameSerializer())
    )
    assert all(
        hasattr(transport, attribute)
        for attribute in ("_client", "_params", "_output_name", "_output")
    )
    output = install_caption_transport(transport, CaptionLedger())
    if closing:
        output._client._closing = True
    assert transport.output() is output
    assert (
        await output._write_frame(
            OutputTransportMessageUrgentFrame(message={"kind": "assistant_text", "text": "Hi"})
        )
        is False
    )


async def test_stale_synthesis_text_is_not_emitted_after_interruption(monkeypatch):
    async def skip_base_process_frame(_self, _frame, _direction):
        pass

    monkeypatch.setattr(FrameProcessor, "process_frame", skip_base_process_frame)
    ledger = CaptionLedger()
    output = _Output(ledger)
    wire = WireEvents(urgent=(TTSTextFrame,))
    outgoing = []

    async def capture(message, direction=None) -> None:
        outgoing.append(message)
        if isinstance(message, CaptionMessageFrame):
            await output.send_message(message)
        elif isinstance(message, InterruptionFrame):
            ledger.interrupt(message.id)

    wire.push_frame = capture
    await wire.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await wire.process_frame(_tts("cancelled", "old"), FrameDirection.DOWNSTREAM)
    await wire.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    await wire.process_frame(_tts("stale after interruption", "old"), FrameDirection.DOWNSTREAM)
    await wire.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await wire.process_frame(_tts("active", "new"), FrameDirection.DOWNSTREAM)
    await wire.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await wire.process_frame(_tts("after response", "new"), FrameDirection.DOWNSTREAM)
    captions = [
        frame.message["text"] for frame in outgoing if isinstance(frame, CaptionMessageFrame)
    ]
    assert captions == ["cancelled", "active"]
