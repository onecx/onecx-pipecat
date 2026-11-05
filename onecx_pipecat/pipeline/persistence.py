"""Wire aggregator events to persistence and bound save-task teardown.

``ReplyTracker`` owns user-turn and reply lifecycle; ``_ContextSync`` coordinates
context updates and save completion across pipeline boundaries.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING

from pipecat.frames.frames import CancelFrame, Frame, InterruptionFrame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from onecx_pipecat.models import VoiceEntry
from onecx_pipecat.pipeline.captions import CaptionLedger
from onecx_pipecat.pipeline.processors import Emit
from onecx_pipecat.pipeline.replies import ReplyTracker, UserTurn

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMAssistantAggregator,
        LLMUserAggregator,
    )


logger = logging.getLogger(__name__)
_TEARDOWN_SAVE_TIMEOUT_SECS = 20


class _ContextSync(FrameProcessor):
    """Wait for interrupted reply reconciliation before passing context frames."""

    def __init__(
        self,
        replies: ReplyTracker,
        *,
        direction: FrameDirection,
        capture: Callable[[LLMContext], None] | None = None,
    ) -> None:
        super().__init__(name=f"ContextSync-{direction.name}")
        self._replies = replies
        self._direction = direction
        self._capture = capture
        self._saves: dict[asyncio.Task[None], tuple[UserTurn | None, str]] = {}
        self._completions: dict[asyncio.Event, tuple[UserTurn | None, str]] = {}

    def save_interrupted(
        self, save: Coroutine[object, object, None], turn: UserTurn | None, name: str
    ) -> None:
        self.track_save(asyncio.create_task(save), turn, name)

    def track_save(self, task: asyncio.Task, turn: UserTurn | None, name: str) -> None:
        self._saves[task] = (turn, name)

        def finish(completed: asyncio.Task) -> None:
            self._saves.pop(completed, None)
            if completed.cancelled():
                logger.debug("Owned assistant persistence task was cancelled")
            elif failure := completed.exception():
                logger.error(
                    "Owned assistant persistence task failed",
                    exc_info=(type(failure), failure, failure.__traceback__),
                )

        task.add_done_callback(finish)

    def track_completion(self, completion: asyncio.Event, turn: UserTurn | None, name: str) -> None:
        self._completions[completion] = (turn, name)

    async def wait_for_saves(self) -> None:
        while self._saves or any(not completion.is_set() for completion in self._completions):
            tasks = list(self._saves)
            completions = [
                completion for completion in self._completions if not completion.is_set()
            ]
            if tasks:
                await asyncio.gather(*(asyncio.shield(task) for task in tasks))
            if completions:
                await asyncio.gather(*(completion.wait() for completion in completions))
        self._completions.clear()

    async def wait_for_prior_saves(self, turn: UserTurn) -> None:
        prior = [task for task, (saved_turn, _) in self._saves.items() if saved_turn is not turn]
        if prior:
            await asyncio.gather(*(asyncio.shield(task) for task in prior))
        prior_completions = [
            completion
            for completion, (saved_turn, _) in self._completions.items()
            if saved_turn is not turn
        ]
        if prior_completions:
            await asyncio.gather(*(completion.wait() for completion in prior_completions))
            for completion in prior_completions:
                self._completions.pop(completion, None)

    async def cleanup(self) -> None:
        try:
            await asyncio.wait_for(self.wait_for_saves(), timeout=_TEARDOWN_SAVE_TIMEOUT_SECS)
        except TimeoutError:
            pending = {
                name
                for completion, (_, name) in self._completions.items()
                if not completion.is_set()
            }
            pending.update(name for _, name in self._saves.values())
            logger.error(
                "Timed out waiting for assistant persistence; pending replies: %s",
                ", ".join(sorted(pending)) or "unknown",
            )
            for task in self._saves:
                task.cancel()
            for completion in self._completions:
                completion.set()
            self._completions.clear()
        finally:
            await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, InterruptionFrame | CancelFrame)
            and direction is FrameDirection.DOWNSTREAM
        ):
            self._replies.begin_reconciliation(frame.id)
        elif isinstance(frame, LLMContextFrame) and direction is self._direction:
            await self._replies.wait_for_reconciliation()
            if self._capture is not None:
                self._capture(frame.context)
        await self.push_frame(frame, direction)


def attach_persistence(
    user: LLMUserAggregator,
    assistant: LLMAssistantAggregator,
    save: Callable[[VoiceEntry], Awaitable[str]],
    emit: Emit,
    *,
    captions: CaptionLedger,
    context: LLMContext,
    fatal_context_failure: Callable[[Exception], Awaitable[None]],
) -> tuple[FrameProcessor, FrameProcessor]:
    replies = ReplyTracker(
        save,
        emit,
        captions=captions,
        context=context,
        fatal_context_failure=fatal_context_failure,
    )
    downstream_sync = _ContextSync(
        replies, direction=FrameDirection.DOWNSTREAM, capture=replies.capture_inference
    )
    upstream_sync = _ContextSync(
        replies, direction=FrameDirection.UPSTREAM, capture=replies.capture_followup
    )
    replies.attach_context_syncs(downstream_sync, upstream_sync)

    user.event_handler("on_before_push_frame")(replies.before_user_push)
    user.event_handler("on_user_turn_started")(replies.on_user_turn_started)
    user.event_handler("on_user_turn_stopped")(replies.on_user_turn_stopped)
    assistant.event_handler("on_assistant_turn_started")(replies.on_assistant_turn_started)
    assistant.event_handler("on_before_process_frame")(replies.before_assistant_frame)
    assistant.event_handler("on_after_process_frame")(replies.after_assistant_frame)
    assistant.event_handler("on_assistant_turn_stopped")(replies.on_assistant_turn_stopped)

    return downstream_sync, upstream_sync
