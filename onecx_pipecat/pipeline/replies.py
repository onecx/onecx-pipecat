"""Tracks user-turn persistence and assistant reply reconciliation."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMFullResponseEndFrame,
)

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

    from onecx_pipecat.pipeline.persistence import _ContextSync


logger = logging.getLogger(__name__)


class UserTurn:
    """Outcome of one user turn's HUMAN save, awaited by its assistant reply."""

    def __init__(self) -> None:
        self.saved: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.failed = False

    def settle(self) -> None:
        if not self.saved.done():
            self.saved.set_result(None)


class AssistantReply:
    def __init__(
        self,
        name: str,
        turn: UserTurn | None,
        anchor: object | None,
        context_messages: list[dict],
    ) -> None:
        self.name = name
        self.turn = turn
        self.anchor = anchor
        self.prior_message_ids = frozenset(id(message) for message in context_messages)
        self.claimed = False
        self.finished = False
        self.completion = asyncio.Event()
        self.finishing_caption = ""
        self.finishing_spoken = ""


class ReplyTracker:
    def __init__(
        self,
        save: Callable[[VoiceEntry], Awaitable[str]],
        emit: Emit,
        *,
        captions: CaptionLedger,
        context: LLMContext,
        fatal_context_failure: Callable[[Exception], Awaitable[None]],
    ) -> None:
        self._save = save
        self._emit = emit
        self._captions = captions
        self._context = context
        self._fatal_context_failure = fatal_context_failure
        self._current: UserTurn | None = None
        self._replies: deque[AssistantReply] = deque()
        self._reply_count = 0
        self._event_reply: ContextVar[AssistantReply | None] = ContextVar(
            "assistant_reply", default=None
        )
        self._interruptions: dict[int, tuple[str, AssistantReply | None]] = {}
        self._finishing: dict[int, AssistantReply] = {}
        self._reconciling: dict[int, asyncio.Event] = {}
        self._downstream_sync: _ContextSync | None = None
        self._upstream_sync: _ContextSync | None = None

    def attach_context_syncs(self, downstream: _ContextSync, upstream: _ContextSync) -> None:
        self._downstream_sync = downstream
        self._upstream_sync = upstream

    def _downstream(self) -> _ContextSync:
        if self._downstream_sync is None:
            raise RuntimeError("reply tracker has no downstream context sync")
        return self._downstream_sync

    def _new_reply(
        self, turn: UserTurn | None, anchor: object | None, messages: list[dict]
    ) -> AssistantReply:
        self._reply_count += 1
        return AssistantReply(f"reply-{self._reply_count}", turn, anchor, messages)

    def capture_inference(self, inference_context: LLMContext) -> None:
        messages = inference_context.get_messages()
        anchor = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            None,
        )
        reply = self._new_reply(self._current, anchor, messages)
        self._replies.append(reply)
        self._downstream().track_completion(reply.completion, reply.turn, reply.name)

    def capture_followup(self, inference_context: LLMContext) -> None:
        parent = self._replies[-1] if self._replies else None
        if parent is None:
            self.capture_inference(inference_context)
            return
        reply = self._new_reply(
            parent.turn,
            parent.anchor,
            inference_context.get_messages(),
        )
        self._replies.append(reply)
        self._downstream().track_completion(reply.completion, reply.turn, reply.name)

    def _oldest_unfinished_reply(self) -> AssistantReply | None:
        return next(
            (reply for reply in self._replies if not reply.claimed and not reply.finished),
            None,
        )

    def _discard_claimed_replies(self) -> None:
        while self._replies and (self._replies[0].claimed or self._replies[0].finished):
            self._replies.popleft()

    async def before_user_push(self, _aggregator: LLMUserAggregator, frame: Frame) -> None:
        if (
            isinstance(frame, InterruptionFrame)
            and frame.broadcast_sibling_id is not None
            and frame.id < frame.broadcast_sibling_id
        ):
            self.begin_reconciliation(frame.id)

    async def on_user_turn_started(self, _aggregator: LLMUserAggregator, _strategy) -> None:
        self._current = UserTurn()

    async def on_user_turn_stopped(self, _aggregator, _strategy, message) -> None:
        if self._current is None:
            self._current = UserTurn()
        turn = self._current
        text = (message.content or "").strip()
        if not text:
            turn.settle()
            return
        try:
            await self.wait_for_reconciliation()
            await self._downstream().wait_for_prior_saves(turn)
            await self._save(VoiceEntry(kind=VoiceEntryKind.HUMAN, text=text))
        except Exception as exc:  # noqa: BLE001 - reported as a typed error, no fabricated history
            turn.failed = True
            await self._emit(ErrorEvent(code="persistence_failed", message=f"human entry: {exc}"))
        finally:
            turn.settle()

    async def on_assistant_turn_started(self, _aggregator: LLMAssistantAggregator) -> None:
        if self._oldest_unfinished_reply() is None:
            reply = self._new_reply(self._current, None, self._context.get_messages())
            self._replies.append(reply)
            self._downstream().track_completion(reply.completion, reply.turn, reply.name)

    async def _save_assistant(
        self, turn: UserTurn | None, text: str, status: VoiceEntryStatus
    ) -> None:
        if turn is not None:
            await turn.saved
        if (turn is not None and turn.failed) or not text:
            return
        try:
            entry_id = await self._save(
                VoiceEntry(kind=VoiceEntryKind.ASSISTANT, text=text, status=status)
            )
        except Exception as exc:  # noqa: BLE001
            await self._emit(
                ErrorEvent(code="persistence_failed", message=f"assistant entry: {exc}")
            )
            return
        await self._emit(TurnComplete(entry_id=entry_id))

    async def _persist_reply(
        self, reply: AssistantReply, text: str, status: VoiceEntryStatus
    ) -> None:
        try:
            await self._save_assistant(reply.turn, text, status)
        finally:
            reply.completion.set()

    async def before_assistant_frame(
        self, aggregator: LLMAssistantAggregator, frame: Frame
    ) -> None:
        reply = self._oldest_unfinished_reply()
        if isinstance(
            frame,
            InterruptionFrame
            | CancelFrame
            | EndFrame
            | LLMFullResponseEndFrame
            | LLMAssistantPushAggregationFrame,
        ):
            self._event_reply.set(reply)
        if isinstance(frame, LLMFullResponseEndFrame | LLMAssistantPushAggregationFrame):
            caption = self._captions.settled.complete(frame)
            if reply is not None:
                self._finishing[frame.id] = reply
                reply.finishing_caption = caption
                reply.finishing_spoken = aggregator.aggregation_string()
        if isinstance(frame, InterruptionFrame | CancelFrame):
            self._interruptions[frame.id] = (
                aggregator.aggregation_string(),
                reply,
            )

    async def after_assistant_frame(
        self, _aggregator: LLMAssistantAggregator, frame: Frame
    ) -> None:
        if reply := self._finishing.pop(frame.id, None):
            reply.finished = True
        if not isinstance(frame, InterruptionFrame | CancelFrame):
            return
        reply: AssistantReply | None = None
        shown = ""
        try:
            spoken, reply = self._interruptions.pop(frame.id)
            shown = self._captions.take_interrupted(frame.id)
            if reply is None:
                return
            for frame_id, completing in list(self._finishing.items()):
                if completing is reply:
                    del self._finishing[frame_id]
            if not reply.claimed and not reply.finished:
                reply.claimed = True
                shown = shown or reply.finishing_caption
                spoken = spoken or reply.finishing_spoken
                if shown or spoken.strip():
                    _replace_reply(
                        self._context,
                        spoken=spoken,
                        shown=shown,
                        anchor=reply.anchor,
                        prior_message_ids=reply.prior_message_ids,
                    )
                if shown:
                    self._downstream().save_interrupted(
                        self._persist_reply(reply, shown, VoiceEntryStatus.INTERRUPTED),
                        reply.turn,
                        reply.name,
                    )
                else:
                    reply.completion.set()
                self._discard_claimed_replies()
        except Exception as exc:  # noqa: BLE001 - the session must not infer from stale context
            logger.exception("Could not reconcile interrupted assistant context")
            if reply is not None and shown and not reply.completion.is_set():
                self._downstream().save_interrupted(
                    self._persist_reply(reply, shown, VoiceEntryStatus.INTERRUPTED),
                    reply.turn,
                    reply.name,
                )
            elif reply is not None:
                reply.completion.set()
            await self._fatal_context_failure(exc)
        finally:
            self.finish_reconciliation(frame.id)

    async def on_assistant_turn_stopped(self, _aggregator: LLMAssistantAggregator, message) -> None:
        if message.interrupted:
            return
        reply = (
            self._event_reply.get()
            or self._oldest_unfinished_reply()
            or self._new_reply(self._current, None, self._context.get_messages())
        )
        if reply.claimed:
            return
        reply.claimed = True
        if task := asyncio.current_task():
            self._downstream().track_save(task, reply.turn, reply.name)
        await self._persist_reply(reply, message.content.strip(), VoiceEntryStatus.COMPLETED)
        self._discard_claimed_replies()

    def begin_reconciliation(self, frame_id: int) -> None:
        self._reconciling.setdefault(frame_id, asyncio.Event())

    def finish_reconciliation(self, frame_id: int) -> None:
        if event := self._reconciling.pop(frame_id, None):
            event.set()

    async def wait_for_reconciliation(self) -> None:
        while self._reconciling:
            await asyncio.gather(*(event.wait() for event in self._reconciling.values()))


def _replace_reply(
    context: LLMContext,
    *,
    spoken: str,
    shown: str,
    anchor: object | None,
    prior_message_ids: frozenset[int] | None = None,
) -> None:
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
        matching = (
            index
            for index in range(boundary - 1, start - 1, -1)
            if isinstance(messages[index], dict)
            and messages[index].get("role") == "assistant"
            and messages[index].get("content") == spoken
            and (prior_message_ids is None or id(messages[index]) not in prior_message_ids)
        )
        index = next(matching, None)
        if index is None and prior_message_ids is not None:
            index = next(
                (
                    candidate
                    for candidate in range(len(messages) - 1, boundary - 1, -1)
                    if isinstance(messages[candidate], dict)
                    and messages[candidate].get("role") == "assistant"
                    and messages[candidate].get("content") == spoken
                    and id(messages[candidate]) not in prior_message_ids
                ),
                None,
            )
        if index is not None:
            message = messages[index]
            replacement = {**message, "content": shown}
            if index < boundary:
                messages[index] = replacement
            else:
                messages.pop(index)
                messages.insert(boundary, replacement)
        else:
            raise RuntimeError("The interrupted assistant text is not in the context")
    else:
        messages.insert(boundary, {"role": "assistant", "content": shown})
    context.set_messages(messages)
