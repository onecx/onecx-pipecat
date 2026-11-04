"""Durable voice entries, written from the context aggregators' turn events.

A user turn is persisted as a HUMAN entry when it ends with non-blank text; an
assistant turn as an ASSISTANT entry, COMPLETED or INTERRUPTED. Captions reach
the client ahead of the audio, so an INTERRUPTED entry holds the sentences that
were emitted, not only the ones that were spoken, and the LLM context is updated
to match. Pipecat starts the LLM before ``on_user_turn_stopped`` fires, so a
HUMAN persistence failure cannot prevent the reply: the error is reported and
the reply of that turn is neither persisted nor acknowledged with
``turn_complete``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from contextvars import ContextVar
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from onecx_pipecat.models import VoiceEntry, VoiceEntryKind, VoiceEntryStatus
from onecx_pipecat.pipeline.captions import CaptionLedger
from onecx_pipecat.pipeline.processors import Emit
from onecx_pipecat.protocol import ErrorEvent, TurnComplete

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMAssistantAggregator,
        LLMUserAggregator,
    )


logger = logging.getLogger(__name__)


class _ContextSync(FrameProcessor):
    """Wait for interrupted context updates without holding control frames."""

    def __init__(
        self,
        captions: CaptionLedger,
        *,
        direction: FrameDirection,
        capture: Callable[[LLMContext], None] | None = None,
    ) -> None:
        super().__init__(name=f"ContextSync-{direction.name}")
        self._captions = captions
        self._direction = direction
        self._capture = capture
        self._saves: dict[asyncio.Task[None], _UserTurn | None] = {}
        self._completions: dict[asyncio.Event, _UserTurn | None] = {}

    def save_interrupted(
        self, save: Coroutine[object, object, None], turn: _UserTurn | None
    ) -> None:
        self.track_save(asyncio.create_task(save), turn)

    def track_save(self, task: asyncio.Task, turn: _UserTurn | None) -> None:
        self._saves[task] = turn
        task.add_done_callback(lambda completed: self._saves.pop(completed, None))

    def track_completion(self, completion: asyncio.Event, turn: _UserTurn | None) -> None:
        self._completions[completion] = turn

    async def wait_for_saves(self) -> None:
        if self._saves:
            await asyncio.gather(*self._saves)

    async def wait_for_prior_saves(self, turn: _UserTurn) -> None:
        prior = [task for task, saved_turn in self._saves.items() if saved_turn is not turn]
        if prior:
            await asyncio.gather(*prior)
        prior_completions = [
            completion
            for completion, saved_turn in self._completions.items()
            if saved_turn is not turn
        ]
        if prior_completions:
            await asyncio.gather(*(completion.wait() for completion in prior_completions))
            for completion in prior_completions:
                self._completions.pop(completion, None)

    async def cleanup(self) -> None:
        await self.wait_for_saves()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, InterruptionFrame | CancelFrame)
            and direction is FrameDirection.DOWNSTREAM
        ):
            self._captions.begin_reconciliation(frame.id)
        elif isinstance(frame, LLMContextFrame) and direction is self._direction:
            await self._captions.wait_for_reconciliation()
            if self._capture is not None:
                self._capture(frame.context)
        await self.push_frame(frame, direction)


class _UserTurn:
    """Outcome of one user turn's HUMAN save, awaited by the assistant turn that answers it."""

    def __init__(self) -> None:
        self.saved: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.failed = False

    def settle(self) -> None:
        if not self.saved.done():
            self.saved.set_result(None)


class _AssistantReply:
    def __init__(self, turn: _UserTurn | None, anchor: object | None) -> None:
        self.turn = turn
        self.anchor = anchor
        self.claimed = False
        self.finished = False
        self.completion: asyncio.Event | None = None
        self.finishing_caption = ""
        self.finishing_spoken = ""


def attach_persistence(
    user: LLMUserAggregator,
    assistant: LLMAssistantAggregator,
    save: Callable[[VoiceEntry], Awaitable[str]],
    emit: Emit,
    *,
    captions: CaptionLedger,
    context: LLMContext,
) -> tuple[FrameProcessor, FrameProcessor]:
    current: _UserTurn | None = None
    active_reply: _AssistantReply | None = None
    event_reply: ContextVar[_AssistantReply | None] = ContextVar("assistant_reply", default=None)
    interruptions: dict[int, tuple[str, _AssistantReply | None]] = {}
    finishing: dict[int, _AssistantReply] = {}

    def capture_inference(inference_context: LLMContext) -> None:
        nonlocal active_reply
        anchor = next(
            (
                message
                for message in reversed(inference_context.get_messages())
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            None,
        )
        active_reply = _AssistantReply(current, anchor)

    def capture_followup(inference_context: LLMContext) -> None:
        nonlocal active_reply
        if active_reply is None:
            capture_inference(inference_context)
        else:
            active_reply = _AssistantReply(active_reply.turn, active_reply.anchor)

    downstream_sync = _ContextSync(
        captions, direction=FrameDirection.DOWNSTREAM, capture=capture_inference
    )
    upstream_sync = _ContextSync(
        captions, direction=FrameDirection.UPSTREAM, capture=capture_followup
    )

    @user.event_handler("on_before_push_frame")
    async def before_user_push(_aggregator, frame: Frame) -> None:
        if (
            isinstance(frame, InterruptionFrame)
            and frame.broadcast_sibling_id is not None
            and frame.id < frame.broadcast_sibling_id
        ):
            # Pipecat creates the downstream interruption before its upstream sibling.
            captions.begin_reconciliation(frame.id)

    @user.event_handler("on_user_turn_started")
    async def on_user_turn_started(_aggregator, _strategy) -> None:
        nonlocal current
        current = _UserTurn()

    @user.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(_aggregator, _strategy, message) -> None:
        nonlocal current
        if current is None:
            current = _UserTurn()
        turn = current
        text = (message.content or "").strip()
        if not text:
            turn.settle()
            return
        try:
            await captions.wait_for_reconciliation()
            await downstream_sync.wait_for_prior_saves(turn)
            await save(VoiceEntry(kind=VoiceEntryKind.HUMAN, text=text))
        except Exception as exc:  # noqa: BLE001 - reported as a typed error, no fabricated history
            turn.failed = True
            await emit(ErrorEvent(code="persistence_failed", message=f"human entry: {exc}"))
        finally:
            turn.settle()

    @assistant.event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(_aggregator) -> None:
        nonlocal active_reply
        if active_reply is None:
            active_reply = _AssistantReply(current, None)

    async def save_assistant(turn: _UserTurn | None, text: str, status: VoiceEntryStatus) -> None:
        if turn is not None:
            await turn.saved
        if (turn is not None and turn.failed) or not text:
            return
        try:
            entry_id = await save(
                VoiceEntry(kind=VoiceEntryKind.ASSISTANT, text=text, status=status)
            )
        except Exception as exc:  # noqa: BLE001
            await emit(ErrorEvent(code="persistence_failed", message=f"assistant entry: {exc}"))
            return
        await emit(TurnComplete(entry_id=entry_id))

    async def persist_reply(reply: _AssistantReply, text: str, status: VoiceEntryStatus) -> None:
        try:
            await save_assistant(reply.turn, text, status)
        finally:
            if reply.completion is not None:
                reply.completion.set()

    @assistant.event_handler("on_before_process_frame")
    async def before_assistant_frame(aggregator, frame: Frame) -> None:
        if isinstance(
            frame,
            InterruptionFrame
            | CancelFrame
            | EndFrame
            | LLMFullResponseEndFrame
            | LLMAssistantPushAggregationFrame,
        ):
            # Pipecat schedules turn callbacks as tasks; preserve their reply association.
            event_reply.set(active_reply)
        if isinstance(frame, LLMFullResponseEndFrame | LLMAssistantPushAggregationFrame):
            caption = captions.settled.complete(frame)
            if active_reply is not None:
                finishing[frame.id] = active_reply
                active_reply.finishing_caption = caption
                active_reply.finishing_spoken = aggregator.aggregation_string()
                if caption or active_reply.finishing_spoken:
                    active_reply.completion = asyncio.Event()
                    downstream_sync.track_completion(active_reply.completion, active_reply.turn)
        if isinstance(frame, InterruptionFrame | CancelFrame):
            if active_reply is not None and active_reply.completion is None:
                active_reply.completion = asyncio.Event()
                downstream_sync.track_completion(active_reply.completion, active_reply.turn)
            interruptions[frame.id] = (aggregator.aggregation_string(), active_reply)

    @assistant.event_handler("on_after_process_frame")
    async def after_assistant_frame(_aggregator, frame: Frame) -> None:
        if reply := finishing.pop(frame.id, None):
            reply.finished = True
        if not isinstance(frame, InterruptionFrame | CancelFrame):
            return
        try:
            spoken, reply = interruptions.pop(frame.id)
            shown = captions.take_interrupted(frame.id)
            if reply is None:
                reply = _AssistantReply(current, None)
            for frame_id, completing in list(finishing.items()):
                if completing is reply:
                    del finishing[frame_id]
            if not reply.claimed and not reply.finished:
                reply.claimed = True
                shown = shown or reply.finishing_caption
                if not spoken and reply.finishing_spoken:
                    spoken = reply.finishing_spoken
                if shown:
                    _replace_reply(context, spoken=spoken, shown=shown, anchor=reply.anchor)
                text = shown or spoken.strip()
                if text:
                    downstream_sync.save_interrupted(
                        persist_reply(reply, text, VoiceEntryStatus.INTERRUPTED), reply.turn
                    )
                elif reply.completion is not None:
                    reply.completion.set()
        except Exception:  # noqa: BLE001 - prevent inference with unreconciled context
            captions.finish_reconciliation(frame.id, failed=True)
            logger.exception("Could not reconcile interrupted assistant context")
            await emit(
                ErrorEvent(
                    code="context_sync_failed",
                    message="Interrupted assistant context could not be reconciled",
                )
            )
        else:
            captions.finish_reconciliation(frame.id)

    @assistant.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(_aggregator, message) -> None:
        if message.interrupted:
            return
        reply = event_reply.get() or active_reply or _AssistantReply(current, None)
        if reply.claimed:
            return
        reply.claimed = True
        if task := asyncio.current_task():
            downstream_sync.track_save(task, reply.turn)
        await persist_reply(reply, message.content.strip(), VoiceEntryStatus.COMPLETED)

    return downstream_sync, upstream_sync


def _replace_reply(context: LLMContext, *, spoken: str, shown: str, anchor: object | None) -> None:
    """Place the emitted text in its user turn without changing tool or later messages."""
    messages = list(context.get_messages())
    start = 0
    if anchor is not None:
        start = next((index + 1 for index, message in enumerate(messages) if message is anchor), -1)
        if start < 0:
            raise RuntimeError("The interrupted turn's user message is no longer in the context")
    boundary = next(
        (
            index
            for index in range(start, len(messages))
            if isinstance(messages[index], dict) and messages[index].get("role") == "user"
        ),
        len(messages),
    )
    if spoken:
        for index in range(len(messages) - 1, start - 1, -1):
            message = messages[index]
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and message.get("content") == spoken
            ):
                replacement = {**message, "content": shown}
                if index < boundary:
                    messages[index] = replacement
                else:
                    messages.pop(index)
                    messages.insert(boundary, replacement)
                break
        else:
            raise RuntimeError("The interrupted assistant text is not in the context")
    else:
        messages.insert(boundary, {"role": "assistant", "content": shown})
    context.set_messages(messages)
