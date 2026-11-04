from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test

from onecx_pipecat.pipeline.captions import CaptionLedger
from onecx_pipecat.protocol import AssistantText


def _shown(text: str) -> OutputTransportMessageUrgentFrame:
    return OutputTransportMessageUrgentFrame(message=AssistantText(text=text).model_dump())


def _reached_transport(text: str) -> AggregatedTextFrame:
    frame = AggregatedTextFrame(text, aggregated_by="sentence")
    frame.will_be_spoken = True
    return frame


async def _run(ledger: CaptionLedger, *frames) -> None:
    class Assistant(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    assistant = Assistant()

    @assistant.event_handler("on_after_process_frame")
    async def after_frame(_assistant, frame: Frame) -> None:
        ledger.settled.complete(frame)

    # Pauses let the data frames settle before the next system frame overtakes them.
    paced = [f for frame in frames for f in (frame, SleepFrame(sleep=0.02))]
    await run_test(Pipeline([ledger.shown, ledger.settled, assistant]), frames_to_send=paced)


async def test_interruption_keeps_every_shown_caption_of_the_playing_reply():
    ledger = CaptionLedger()
    await _run(
        ledger,
        _shown("Sentence 1."),
        _shown("Sentence 2."),
        _shown("Sentence 3."),
        _reached_transport("Sentence 1."),
        InterruptionFrame(),
    )
    assert ledger.take_interrupted() == "Sentence 1. Sentence 2. Sentence 3."
    assert ledger.take_interrupted() == ""


async def test_interruption_skips_replies_that_finished_playing():
    ledger = CaptionLedger()
    await _run(
        ledger,
        _shown("Let me check."),
        _shown("It is sunny."),
        _reached_transport("Let me check."),
        LLMFullResponseEndFrame(),
        InterruptionFrame(),
    )
    assert ledger.take_interrupted() == "It is sunny."


def test_interrupted_snapshots_survive_later_captions_and_interruptions():
    ledger = CaptionLedger()
    ledger.record("First reply.")
    ledger.interrupt()
    ledger.record("Second reply.")
    ledger.interrupt()
    assert ledger.take_interrupted() == "First reply."
    assert ledger.take_interrupted() == "Second reply."
    assert ledger.take_interrupted() == ""


def test_interrupted_snapshots_are_matched_to_their_interruption():
    ledger = CaptionLedger()
    ledger.record("First reply.")
    ledger.interrupt(1)
    ledger.record("Second reply.")
    ledger.interrupt(2)
    assert ledger.take_interrupted(2) == "Second reply."
    assert ledger.take_interrupted(1) == "First reply."
