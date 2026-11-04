from types import SimpleNamespace

import pytest
from pipecat.frames.frames import InterruptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from onecx_pipecat.models import VoiceEntryKind, VoiceEntryStatus
from onecx_pipecat.pipeline.captions import CaptionLedger
from onecx_pipecat.pipeline.persistence import _replace_reply, attach_persistence
from tests.fakes import FakePersistence


class _Aggregator:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.spoken = ""

    def event_handler(self, name):
        def register(fn):
            self.handlers[name] = fn
            return fn

        return register

    def aggregation_string(self) -> str:
        return self.spoken


class _Harness:
    def __init__(self, store: FakePersistence) -> None:
        self.user, self.assistant = _Aggregator(), _Aggregator()
        self.emitted: list = []
        self.store = store
        self.captions = CaptionLedger()
        self.context = LLMContext(messages=[{"role": "system", "content": "prompt"}])

        async def emit(event) -> None:
            self.emitted.append(event)

        async def save(entry) -> str:
            return await store.append("s1", entry)

        async def fatal_context_failure(failure: Exception) -> None:
            raise failure

        self.context_sync, _ = attach_persistence(
            self.user,
            self.assistant,
            save,
            emit,
            captions=self.captions,
            context=self.context,
            fatal_context_failure=fatal_context_failure,
        )

    async def user_said(self, content) -> None:
        await self.user.handlers["on_user_turn_started"](None, None)
        await self.user.handlers["on_user_turn_stopped"](
            None, None, SimpleNamespace(content=content)
        )

    async def assistant_said(self, content: str, interrupted: bool = False) -> None:
        await self.assistant.handlers["on_assistant_turn_started"](None)
        if interrupted:
            frame = InterruptionFrame()
            self.captions.interrupt(frame.id)
            self.assistant.spoken = content
            await self.assistant.handlers["on_before_process_frame"](self.assistant, frame)
            if content:
                self.context.add_message({"role": "assistant", "content": content})
            await self.assistant.handlers["on_after_process_frame"](self.assistant, frame)
            await self.context_sync.wait_for_saves()
            return
        if content:
            # The aggregator adds the spoken text to the context before the event.
            self.context.add_message({"role": "assistant", "content": content})
        await self.assistant.handlers["on_assistant_turn_stopped"](
            None, SimpleNamespace(content=content, interrupted=interrupted)
        )


async def test_completed_turn_persists_human_then_assistant_and_acknowledges():
    h = _Harness(FakePersistence())
    await h.user_said("  hello ")
    await h.assistant_said("hi there")
    assert [(e.kind, e.text, e.status) for e in h.store.entries] == [
        (VoiceEntryKind.HUMAN, "hello", VoiceEntryStatus.COMPLETED),
        (VoiceEntryKind.ASSISTANT, "hi there", VoiceEntryStatus.COMPLETED),
    ]
    assert [(e.kind, e.entry_id) for e in h.emitted] == [("turn_complete", "s1:2")]


async def test_interrupted_assistant_turn_is_persisted_as_interrupted():
    h = _Harness(FakePersistence())
    await h.user_said("hello")
    h.captions.record("partial")
    await h.assistant_said("partial", interrupted=True)
    assert (h.store.entries[-1].text, h.store.entries[-1].status) == (
        "partial",
        VoiceEntryStatus.INTERRUPTED,
    )
    assert [e.kind for e in h.emitted] == ["turn_complete"]


@pytest.mark.parametrize("spoken", ["Sentence 1.", ""])
async def test_interrupted_turn_keeps_the_shown_text_not_only_the_spoken(spoken):
    h = _Harness(FakePersistence())
    await h.user_said("hello")
    for caption in ("Sentence 1.", "Sentence 2."):
        h.captions.record(caption)
    await h.assistant_said(spoken, interrupted=True)
    shown = "Sentence 1. Sentence 2."
    assert (h.store.entries[-1].text, h.store.entries[-1].status) == (
        shown,
        VoiceEntryStatus.INTERRUPTED,
    )
    assert h.context.get_messages()[1:] == [{"role": "assistant", "content": shown}]


async def test_interrupted_context_update_preserves_later_user_and_tool_messages():
    context = LLMContext(messages=[{"role": "system", "content": "prompt"}])
    anchor = {"role": "user", "content": "hello"}
    context.add_message(anchor)
    spoken = {"role": "assistant", "content": "Heard sentence.", "name": "bot"}
    context.add_message(spoken)
    next_user = {"role": "user", "content": "Next question"}
    context.add_message(next_user)
    tool_call = {"role": "assistant", "tool_calls": [{"id": "call-1"}]}
    context.add_message(tool_call)

    _replace_reply(
        context, spoken="Heard sentence.", shown="Heard sentence. Also shown.", anchor=anchor
    )

    assert context.get_messages()[1:] == [
        anchor,
        {"role": "assistant", "content": "Heard sentence. Also shown.", "name": "bot"},
        next_user,
        tool_call,
    ]


async def test_interrupted_context_does_not_replace_matching_text_in_a_later_turn():
    context = LLMContext(
        messages=[
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "same text"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "same text"},
        ]
    )
    anchor = context.get_messages()[1]

    _replace_reply(context, spoken="same text", shown="first caption", anchor=anchor)

    assert context.get_messages()[2:] == [
        {"role": "assistant", "content": "first caption"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "same text"},
    ]


@pytest.mark.parametrize("spoken", ["", "Heard sentence."])
async def test_interrupted_context_is_inserted_before_the_next_user_after_tools(spoken):
    context = LLMContext(messages=[{"role": "system", "content": "prompt"}])
    anchor = {"role": "user", "content": "hello"}
    tool_call = {"role": "assistant", "tool_calls": [{"id": "call-1"}]}
    tool_result = {"role": "tool", "tool_call_id": "call-1", "content": "done"}
    next_user = {"role": "user", "content": "Next question"}
    context.add_messages([anchor, tool_call, tool_result, next_user])
    prior_message_ids = frozenset(id(message) for message in context.get_messages())
    if spoken:
        context.add_message({"role": "assistant", "content": spoken})

    _replace_reply(
        context,
        spoken=spoken,
        shown="The full caption.",
        anchor=anchor,
        prior_message_ids=prior_message_ids,
    )

    assert context.get_messages()[1:] == [
        anchor,
        tool_call,
        tool_result,
        {"role": "assistant", "content": "The full caption."},
        next_user,
    ]


async def test_blank_turns_stay_ephemeral():
    h = _Harness(FakePersistence())
    await h.user_said("   ")
    await h.user_said(None)
    await h.assistant_said("  ")
    assert h.store.entries == []
    assert h.emitted == []


async def test_human_persistence_failure_does_not_skip_the_next_turn_after_llm_error():
    store = FakePersistence(fail_human=True)
    h = _Harness(store)
    await h.user_said("hello")
    # The failed LLM turn has no assistant-stopped event to consume the failure.
    assert [(e.kind, e.code) for e in h.emitted] == [("error", "persistence_failed")]
    store.fail_human = False
    await h.user_said("again")
    await h.assistant_said("answer 2")
    assert [e.kind for e in store.entries] == [VoiceEntryKind.HUMAN, VoiceEntryKind.ASSISTANT]
    assert [e.kind for e in h.emitted].count("turn_complete") == 1


async def test_assistant_persistence_failure_emits_error_without_turn_complete():
    h = _Harness(FakePersistence(fail_assistant=True))
    await h.user_said("hello")
    await h.assistant_said("answer")
    assert [e.kind for e in h.store.entries] == [VoiceEntryKind.HUMAN]
    assert [(e.kind, e.code) for e in h.emitted] == [("error", "persistence_failed")]
