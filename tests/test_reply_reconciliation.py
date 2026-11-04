import asyncio
import copy
import threading

from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator

from onecx_pipecat import session as session_module
from onecx_pipecat.app import create_app
from onecx_pipecat.models import VoiceEntryKind, VoiceEntryStatus
from onecx_pipecat.services.inmemory import InMemoryLLM
from tests.fakes import FakePersistence, fake_container
from tests.voice_helpers import (
    GatedTTS,
    _receive_json_bounded,
    _speak,
    _start,
    _until,
    end_session,
)


def test_interruption_after_reply_finishes_does_not_duplicate_it(monkeypatch):
    release_end = threading.Event()
    end_waiting = threading.Event()
    original_stop = LLMAssistantAggregator._trigger_assistant_turn_stopped

    async def delayed_end(self, *, interrupted=False):
        await original_stop(self, interrupted=interrupted)
        if not interrupted:
            end_waiting.set()
            await asyncio.to_thread(release_end.wait)

    monkeypatch.setattr(LLMAssistantAggregator, "_trigger_assistant_turn_stopped", delayed_end)
    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="Finished reply.")
    )
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            _until(ws, "assistant_text")
            assert end_waiting.wait(timeout=3)
            ws.send_json({"type": "user_started_speaking"})
            _until(ws, "interrupted")
            release_end.set()
            end_session(ws)
        finally:
            release_end.set()

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.COMPLETED, "Finished reply.")
    ]


def test_delayed_completion_callback_does_not_turn_finished_reply_into_interruption(monkeypatch):
    release_handler = threading.Event()
    handler_waiting = threading.Event()
    end_processed = threading.Event()
    original_handler = LLMAssistantAggregator._run_handler

    async def delayed_handler(self, event_name, handler, *args, **kwargs):
        if event_name == "on_assistant_turn_stopped" and not args[0].interrupted:
            handler_waiting.set()
            await asyncio.to_thread(release_handler.wait)
        await original_handler(self, event_name, handler, *args, **kwargs)
        if event_name == "on_after_process_frame" and isinstance(args[0], LLMFullResponseEndFrame):
            end_processed.set()

    monkeypatch.setattr(LLMAssistantAggregator, "_run_handler", delayed_handler)
    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="Finished reply.")
    )
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            _until(ws, "assistant_text")
            assert handler_waiting.wait(timeout=3)
            assert end_processed.wait(timeout=3)
            ws.send_json({"type": "user_started_speaking"})
            _until(ws, "interrupted")
            release_handler.set()
            _until(ws, "turn_complete")
            end_session(ws)
        finally:
            release_handler.set()

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.COMPLETED, "Finished reply.")
    ]


def test_next_user_entry_waits_for_delayed_completion_callback(monkeypatch):
    release_handler = threading.Event()
    handler_waiting = threading.Event()
    end_processed = threading.Event()
    original_handler = LLMAssistantAggregator._run_handler

    async def delayed_handler(self, event_name, handler, *args, **kwargs):
        if event_name == "on_assistant_turn_stopped" and args[0].content == "Reply 1.":
            handler_waiting.set()
            await asyncio.to_thread(release_handler.wait)
        await original_handler(self, event_name, handler, *args, **kwargs)
        if event_name == "on_after_process_frame" and isinstance(args[0], LLMFullResponseEndFrame):
            end_processed.set()

    class RecordingLLM(InMemoryLLM):
        def __init__(self):
            super().__init__()
            self.turn = 0

        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                self.turn += 1
                self._reply = f"Reply {self.turn}."
            await super().process_frame(frame, direction)

    monkeypatch.setattr(LLMAssistantAggregator, "_run_handler", delayed_handler)
    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            _until(ws, "assistant_text")
            assert handler_waiting.wait(timeout=3)
            assert end_processed.wait(timeout=3)
            _speak(ws)
            _until(ws, "assistant_text")
            assert [entry.kind for entry in persistence.entries] == [VoiceEntryKind.HUMAN]
            release_handler.set()
            completed = 0
            while completed < 2:
                completed += _receive_json_bounded(ws)["kind"] == "turn_complete"
        finally:
            release_handler.set()
        end_session(ws)

    assert [entry.kind for entry in persistence.entries] == [
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
    ]


def test_next_inference_contains_the_interrupted_caption_before_the_next_user_turn(monkeypatch):
    contexts = []

    class RecordingLLM(InMemoryLLM):
        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                contexts.append(copy.deepcopy(frame.context.get_messages()))
                self._reply = f"Reply {len(contexts)}."
            await super().process_frame(frame, direction)

    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: GatedTTS(
            sample_rate=settings.speech.sample_rate_hz, audio_seconds=1
        ),
    )
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _speak(ws)
        _until(ws, "assistant_text")
        _speak(ws)
        completions = 0
        while completions < 2:
            if _receive_json_bounded(ws)["kind"] == "turn_complete":
                completions += 1
        end_session(ws)

    assert len(contexts) == 2
    assert [(m["role"], m.get("content")) for m in contexts[1]] == [
        ("system", contexts[1][0]["content"]),
        ("user", "in-memory transcript"),
        ("assistant", "Reply 1."),
        ("user", "in-memory transcript"),
    ]
    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.INTERRUPTED, "Reply 1."),
        (VoiceEntryStatus.COMPLETED, "Reply 2."),
    ]


def test_next_inference_has_interrupted_context_even_if_stop_handler_is_delayed(monkeypatch):
    release_handler = threading.Event()
    handler_waiting = threading.Event()
    second_inference = threading.Event()
    contexts = []
    original_handler = LLMAssistantAggregator._run_handler

    async def delayed_handler(self, event_name, handler, *args, **kwargs):
        if event_name == "on_assistant_turn_stopped" and args[0].interrupted:
            handler_waiting.set()
            await asyncio.to_thread(release_handler.wait)
        await original_handler(self, event_name, handler, *args, **kwargs)

    class RecordingLLM(InMemoryLLM):
        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                contexts.append(copy.deepcopy(frame.context.get_messages()))
                self._reply = f"Reply {len(contexts)}."
                if len(contexts) == 2:
                    second_inference.set()
            await super().process_frame(frame, direction)

    monkeypatch.setattr(LLMAssistantAggregator, "_run_handler", delayed_handler)
    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: GatedTTS(
            sample_rate=settings.speech.sample_rate_hz, audio_seconds=1
        ),
    )
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            _until(ws, "assistant_text")
            _speak(ws)
            assert handler_waiting.wait(timeout=3)
            while True:
                event = _receive_json_bounded(ws)
                if event["kind"] == "transcript" and event["state"] == "FINAL":
                    break
            assert second_inference.wait(timeout=3)
            release_handler.set()
            _until(ws, "assistant_text")
        finally:
            release_handler.set()
        end_session(ws)

    assert [(m["role"], m.get("content")) for m in contexts[1][1:]] == [
        ("user", "in-memory transcript"),
        ("assistant", "Reply 1."),
        ("user", "in-memory transcript"),
    ]


def test_next_inference_waits_until_interruption_is_reconciled(monkeypatch):
    release_interruption = threading.Event()
    interruption_waiting = threading.Event()
    second_inference = threading.Event()
    contexts = []
    original_process = LLMAssistantAggregator.process_frame
    interruption_count = 0

    async def delayed_interruption(self, frame, direction):
        nonlocal interruption_count
        if isinstance(frame, InterruptionFrame):
            interruption_count += 1
            if interruption_count == 2:
                interruption_waiting.set()
                await asyncio.to_thread(release_interruption.wait)
        await original_process(self, frame, direction)

    class RecordingLLM(InMemoryLLM):
        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                contexts.append(copy.deepcopy(frame.context.get_messages()))
                self._reply = f"Reply {len(contexts)}."
                if len(contexts) == 2:
                    second_inference.set()
            await super().process_frame(frame, direction)

    monkeypatch.setattr(LLMAssistantAggregator, "process_frame", delayed_interruption)
    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: GatedTTS(
            sample_rate=settings.speech.sample_rate_hz, audio_seconds=1
        ),
    )
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            _until(ws, "assistant_text")
            _speak(ws)
            assert interruption_waiting.wait(timeout=3)
            while True:
                event = _receive_json_bounded(ws)
                if event["kind"] == "transcript" and event["state"] == "FINAL":
                    break
            assert not second_inference.is_set()
            release_interruption.set()
            _until(ws, "assistant_text")
            assert second_inference.is_set()
        finally:
            release_interruption.set()
        end_session(ws)

    assert [(m["role"], m.get("content")) for m in contexts[1][1:]] == [
        ("user", "in-memory transcript"),
        ("assistant", "Reply 1."),
        ("user", "in-memory transcript"),
    ]


def test_next_user_entry_waits_for_interrupted_reply_to_be_saved(monkeypatch):
    release_save = threading.Event()
    save_waiting = threading.Event()

    class SlowAssistantPersistence(FakePersistence):
        async def append(self, session_id, entry, **kwargs):
            if (
                entry.kind is VoiceEntryKind.ASSISTANT
                and entry.status is VoiceEntryStatus.INTERRUPTED
            ):
                save_waiting.set()
                await asyncio.to_thread(release_save.wait)
            return await super().append(session_id, entry, **kwargs)

    class RecordingLLM(InMemoryLLM):
        def __init__(self):
            super().__init__()
            self.turn = 0

        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                self.turn += 1
                self._reply = f"Reply {self.turn}."
            await super().process_frame(frame, direction)

    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: GatedTTS(
            sample_rate=settings.speech.sample_rate_hz, audio_seconds=1
        ),
    )
    container = fake_container(token="t-1")
    persistence = SlowAssistantPersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            _until(ws, "assistant_text")
            _speak(ws)
            _until(ws, "assistant_text")
            assert save_waiting.wait(timeout=3)
            assert [entry.kind for entry in persistence.entries] == [VoiceEntryKind.HUMAN]
            release_save.set()
            completed = 0
            while completed < 2:
                completed += _receive_json_bounded(ws)["kind"] == "turn_complete"
            end_session(ws)
        finally:
            release_save.set()

    assert [entry.kind for entry in persistence.entries] == [
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
    ]


def test_rapid_interruptions_keep_each_reply_with_its_user_turn(monkeypatch):
    contexts = []

    class RecordingLLM(InMemoryLLM):
        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                contexts.append(copy.deepcopy(frame.context.get_messages()))
                self._reply = f"Reply {len(contexts)}."
            await super().process_frame(frame, direction)

    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: GatedTTS(
            sample_rate=settings.speech.sample_rate_hz, audio_seconds=1
        ),
    )
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        events = []
        for number in range(1, 4):
            _speak(ws)
            events.extend(_until(ws, "assistant_text"))
            assert events[-1]["text"] == f"Reply {number}."
        while sum(event["kind"] == "turn_complete" for event in events) < 3:
            events.append(_receive_json_bounded(ws))
        end_session(ws)

    assert [(m["role"], m.get("content")) for m in contexts[2][1:]] == [
        ("user", "in-memory transcript"),
        ("assistant", "Reply 1."),
        ("user", "in-memory transcript"),
        ("assistant", "Reply 2."),
        ("user", "in-memory transcript"),
    ]
    assert [(entry.kind, entry.status, entry.text) for entry in persistence.entries] == [
        (VoiceEntryKind.HUMAN, VoiceEntryStatus.COMPLETED, "in-memory transcript"),
        (VoiceEntryKind.ASSISTANT, VoiceEntryStatus.INTERRUPTED, "Reply 1."),
        (VoiceEntryKind.HUMAN, VoiceEntryStatus.COMPLETED, "in-memory transcript"),
        (VoiceEntryKind.ASSISTANT, VoiceEntryStatus.INTERRUPTED, "Reply 2."),
        (VoiceEntryKind.HUMAN, VoiceEntryStatus.COMPLETED, "in-memory transcript"),
        (VoiceEntryKind.ASSISTANT, VoiceEntryStatus.COMPLETED, "Reply 3."),
    ]


def test_tool_followup_can_stream_and_be_interrupted(monkeypatch):
    contexts = []
    tool_call = {"role": "assistant", "tool_calls": [{"id": "call-1"}]}
    tool_result = {"role": "tool", "tool_call_id": "call-1", "content": "done"}

    class ToolFollowupLLM(InMemoryLLM):
        async def process_frame(self, frame, direction):
            if not isinstance(frame, LLMContextFrame):
                await super().process_frame(frame, direction)
                return
            contexts.append(copy.deepcopy(frame.context.get_messages()))
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame("Checking." if len(contexts) == 1 else "Done."))
            await self.push_frame(LLMFullResponseEndFrame())
            if len(contexts) == 1:
                await self.push_frame(
                    LLMMessagesAppendFrame(messages=[tool_call, tool_result], run_llm=True)
                )

    monkeypatch.setattr(session_module, "build_llm", lambda *args: ToolFollowupLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: GatedTTS(
            sample_rate=settings.speech.sample_rate_hz, audio_seconds=1
        ),
    )
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _speak(ws)
        captions, completed = [], 0
        while captions != ["Checking.", "Done."]:
            event = _receive_json_bounded(ws)
            if event["kind"] == "assistant_text":
                captions.append(event["text"])
            completed += event["kind"] == "turn_complete"
        ws.send_json({"type": "user_started_speaking"})
        while completed < 2:
            completed += _receive_json_bounded(ws)["kind"] == "turn_complete"
        end_session(ws)

    assert len(contexts) == 2
    assert tool_call in contexts[1] and tool_result in contexts[1]
    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.COMPLETED, "Checking."),
        (VoiceEntryStatus.INTERRUPTED, "Done."),
    ]
