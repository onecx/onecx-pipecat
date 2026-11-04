import asyncio
import base64
import copy
import threading

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator

from onecx_pipecat import session as session_module
from onecx_pipecat.app import create_app
from onecx_pipecat.models import VoiceEntryKind, VoiceEntryStatus
from onecx_pipecat.services.inmemory import InMemoryLLM, InMemoryTTS
from tests.fakes import FakePersistence, fake_container

_AUDIO = base64.b64encode(b"\x01\x00" * 160).decode("ascii")


_CHUNKS, _CHUNK_MS = 3, 100


@pytest.fixture
def slow_tts(monkeypatch):
    # A long multi-sentence reply, played in real time, gives the client time to
    # barge in after the first sentence.
    monkeypatch.setattr(
        session_module,
        "build_llm",
        lambda *args: InMemoryLLM(reply=" ".join(f"Sentence {i}." for i in range(30))),
    )
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: InMemoryTTS(
            sample_rate=settings.speech.sample_rate_hz, chunks=_CHUNKS, chunk_ms=_CHUNK_MS
        ),
    )


def _until(ws, kind: str) -> list[dict]:
    events = []
    while True:
        events.append(ws.receive_json())
        if events[-1]["kind"] == kind:
            return events


def test_barge_in_interrupts_the_bot_and_persists_an_interrupted_entry(slow_tts):
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(
            {
                "type": "start_session",
                "agent_id": "onecx-agent",
                "chat_id": "chat-1",
                "apm_principal_token": "t-1",
            }
        )
        assert ws.receive_json()["kind"] == "session_ready"
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})
        _until(ws, "assistant_text")

        ws.send_json({"type": "user_started_speaking"})
        events = _until(ws, "turn_complete")
        assert "interrupted" in [e["kind"] for e in events]

        ws.send_json({"type": "stop"})
        _until(ws, "session_closed")

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [e.status for e in assistant] == [VoiceEntryStatus.INTERRUPTED]


def _sentence_bytes(container) -> int:
    return _CHUNKS * container.settings.speech.sample_rate_hz * _CHUNK_MS // 1000 * 2


def _audio_bytes(event: dict) -> int:
    return len(base64.b64decode(event["audio"])) if event["kind"] == "assistant_audio" else 0


def test_each_caption_reaches_the_client_before_its_audio_has_played(slow_tts):
    container = fake_container(token="t-1")
    sentence = _sentence_bytes(container)
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _speak(ws)
        captions, audio = [], 0
        while audio < 3 * sentence:
            event = ws.receive_json()
            audio += _audio_bytes(event)
            if event["kind"] == "assistant_text":
                assert audio <= len(captions) * sentence
                captions.append(event["text"])
        assert captions[:3] == ["Sentence 0.", "Sentence 1.", "Sentence 2."]

        ws.send_json({"type": "stop"})
        _until(ws, "session_closed")


def test_caption_arrives_while_tts_is_still_streaming(monkeypatch):
    release = threading.Event()
    finished = threading.Event()

    class StreamingTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)
            await asyncio.to_thread(release.wait)
            finished.set()

    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="One sentence.")
    )
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: StreamingTTS(sample_rate=settings.speech.sample_rate_hz),
    )
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            while True:
                event = ws.receive_json()
                if event["kind"] in {"assistant_text", "assistant_audio"}:
                    break
            assert event["kind"] == "assistant_text"
            assert event["text"] == "One sentence."
            assert not finished.is_set()
            assert _until(ws, "assistant_audio")[-1]["kind"] == "assistant_audio"
        finally:
            release.set()


def test_barge_in_after_caption_before_any_audio_keeps_the_caption(monkeypatch):
    release = threading.Event()

    class DelayedTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            await asyncio.to_thread(release.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="Visible before audio.")
    )
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedTTS(sample_rate=settings.speech.sample_rate_hz),
    )
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            events = _until(ws, "assistant_text")
            assert not any(e["kind"] == "assistant_audio" for e in events)
            ws.send_json({"type": "user_started_speaking"})
            events += _until(ws, "turn_complete")
            assert not any(e["kind"] == "assistant_audio" for e in events)
        finally:
            release.set()

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.INTERRUPTED, "Visible before audio.")
    ]


def test_barge_in_keeps_caption_if_assistant_aggregator_has_not_started(monkeypatch):
    release_start = threading.Event()
    release_tts = threading.Event()
    start_waiting = threading.Event()
    persisted = threading.Event()
    original_process = LLMAssistantAggregator.process_frame

    async def delayed_start(self, frame, direction):
        if isinstance(frame, LLMFullResponseStartFrame):
            start_waiting.set()
            await asyncio.to_thread(release_start.wait)
        await original_process(self, frame, direction)

    class SignalingPersistence(FakePersistence):
        async def append(self, session_id, entry, **kwargs):
            entry_id = await super().append(session_id, entry, **kwargs)
            if entry.kind is VoiceEntryKind.ASSISTANT:
                persisted.set()
            return entry_id

    class DelayedTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            await asyncio.to_thread(release_tts.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    monkeypatch.setattr(LLMAssistantAggregator, "process_frame", delayed_start)
    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="Caption before start.")
    )
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedTTS(sample_rate=settings.speech.sample_rate_hz),
    )
    container = fake_container(token="t-1")
    persistence = SignalingPersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            _until(ws, "assistant_text")
            assert start_waiting.wait(timeout=3)
            ws.send_json({"type": "user_started_speaking"})
            _until(ws, "interrupted")
            assert persisted.wait(timeout=2)
        finally:
            release_start.set()
            release_tts.set()

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.INTERRUPTED, "Caption before start.")
    ]


@pytest.mark.parametrize("ending", ["stop", "disconnect"])
def test_ending_session_persists_caption_before_audio(monkeypatch, ending):
    release_tts = threading.Event()
    persisted = threading.Event()

    class DelayedTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            await asyncio.to_thread(release_tts.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    class SlowPersistence(FakePersistence):
        async def append(self, session_id, entry, **kwargs):
            if entry.kind is VoiceEntryKind.ASSISTANT:
                await asyncio.sleep(0.05)
            entry_id = await super().append(session_id, entry, **kwargs)
            if entry.kind is VoiceEntryKind.ASSISTANT:
                persisted.set()
            return entry_id

    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="Caption before stop.")
    )
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedTTS(sample_rate=settings.speech.sample_rate_hz),
    )
    container = fake_container(token="t-1")
    persistence = SlowPersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            _until(ws, "assistant_text")
            if ending == "stop":
                ws.send_json({"type": "stop"})
                _until(ws, "session_closed")
            else:
                ws.close()
                assert persisted.wait(timeout=3)
        finally:
            release_tts.set()

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.INTERRUPTED, "Caption before stop.")
    ]


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
            ws.send_json({"type": "stop"})
            _until(ws, "session_closed")
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
            ws.send_json({"type": "stop"})
            _until(ws, "session_closed")
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
                completed += ws.receive_json()["kind"] == "turn_complete"
        finally:
            release_handler.set()

    assert [entry.kind for entry in persistence.entries] == [
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
    ]


def test_next_inference_contains_the_interrupted_caption_before_the_next_user_turn(monkeypatch):
    release = threading.Event()
    contexts = []

    class RecordingLLM(InMemoryLLM):
        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                contexts.append(copy.deepcopy(frame.context.get_messages()))
                self._reply = f"Reply {len(contexts)}."
            await super().process_frame(frame, direction)

    class DelayedFirstTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            if text == "Reply 1.":
                await asyncio.to_thread(release.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedFirstTTS(sample_rate=settings.speech.sample_rate_hz),
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
            _speak(ws)
            completions = 0
            while completions < 2:
                if ws.receive_json()["kind"] == "turn_complete":
                    completions += 1
            ws.send_json({"type": "stop"})
            _until(ws, "session_closed")
        finally:
            release.set()

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
    release_tts = threading.Event()
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

    class DelayedFirstTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            if text == "Reply 1.":
                await asyncio.to_thread(release_tts.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    monkeypatch.setattr(LLMAssistantAggregator, "_run_handler", delayed_handler)
    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedFirstTTS(sample_rate=settings.speech.sample_rate_hz),
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
                event = ws.receive_json()
                if event["kind"] == "transcript" and event["state"] == "FINAL":
                    break
            assert second_inference.wait(timeout=3)
            release_handler.set()
            _until(ws, "assistant_text")
        finally:
            release_handler.set()
            release_tts.set()

    assert [(m["role"], m.get("content")) for m in contexts[1][1:]] == [
        ("user", "in-memory transcript"),
        ("assistant", "Reply 1."),
        ("user", "in-memory transcript"),
    ]


def test_next_inference_waits_until_interruption_is_reconciled(monkeypatch):
    release_tts = threading.Event()
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

    class DelayedFirstTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            if text == "Reply 1.":
                await asyncio.to_thread(release_tts.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    monkeypatch.setattr(LLMAssistantAggregator, "process_frame", delayed_interruption)
    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedFirstTTS(sample_rate=settings.speech.sample_rate_hz),
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
                event = ws.receive_json()
                if event["kind"] == "transcript" and event["state"] == "FINAL":
                    break
            assert not second_inference.wait(timeout=0.25)
            release_interruption.set()
            _until(ws, "assistant_text")
            assert second_inference.is_set()
        finally:
            release_interruption.set()
            release_tts.set()

    assert [(m["role"], m.get("content")) for m in contexts[1][1:]] == [
        ("user", "in-memory transcript"),
        ("assistant", "Reply 1."),
        ("user", "in-memory transcript"),
    ]


def test_next_user_entry_waits_for_interrupted_reply_to_be_saved(monkeypatch):
    release_tts = threading.Event()
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

    class DelayedFirstTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            if text == "Reply 1.":
                await asyncio.to_thread(release_tts.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedFirstTTS(sample_rate=settings.speech.sample_rate_hz),
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
                completed += ws.receive_json()["kind"] == "turn_complete"
        finally:
            release_save.set()
            release_tts.set()

    assert [entry.kind for entry in persistence.entries] == [
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
    ]


def test_rapid_interruptions_keep_each_reply_with_its_user_turn(monkeypatch):
    release_tts = threading.Event()
    contexts = []

    class RecordingLLM(InMemoryLLM):
        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                contexts.append(copy.deepcopy(frame.context.get_messages()))
                self._reply = f"Reply {len(contexts)}."
            await super().process_frame(frame, direction)

    class DelayedTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            if text in {"Reply 1.", "Reply 2."}:
                await asyncio.to_thread(release_tts.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedTTS(sample_rate=settings.speech.sample_rate_hz),
    )
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            events = []
            for number in range(1, 4):
                _speak(ws)
                events.extend(_until(ws, "assistant_text"))
                assert events[-1]["text"] == f"Reply {number}."
            while sum(event["kind"] == "turn_complete" for event in events) < 3:
                events.append(ws.receive_json())
        finally:
            release_tts.set()

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
    release = threading.Event()
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

    class DelayedFollowupTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            if text == "Done.":
                await asyncio.to_thread(release.wait)
            yield TTSAudioRawFrame(b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id)

    monkeypatch.setattr(session_module, "build_llm", lambda *args: ToolFollowupLLM())
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: DelayedFollowupTTS(sample_rate=settings.speech.sample_rate_hz),
    )
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            captions, completed = [], 0
            while captions != ["Checking.", "Done."]:
                event = ws.receive_json()
                if event["kind"] == "assistant_text":
                    captions.append(event["text"])
                completed += event["kind"] == "turn_complete"
            ws.send_json({"type": "user_started_speaking"})
            while completed < 2:
                completed += ws.receive_json()["kind"] == "turn_complete"
        finally:
            release.set()

    assert len(contexts) == 2
    assert tool_call in contexts[1] and tool_result in contexts[1]
    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.COMPLETED, "Checking."),
        (VoiceEntryStatus.INTERRUPTED, "Done."),
    ]


def test_barge_in_persists_every_caption_shown_not_only_the_spoken_ones(slow_tts):
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    sentence = _sentence_bytes(container)
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _speak(ws)
        events = _until(ws, "assistant_audio")
        ws.send_json({"type": "user_started_speaking"})
        events += _until(ws, "turn_complete")
        assert "interrupted" in [e["kind"] for e in events]

        ws.send_json({"type": "stop"})
        _until(ws, "session_closed")

    captions = [e["text"] for e in events if e["kind"] == "assistant_text"]
    played = sum(_audio_bytes(e) for e in events) // sentence
    assert len(captions) > played + 1
    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.INTERRUPTED, " ".join(captions))
    ]


def _start(ws) -> None:
    ws.send_json(
        {
            "type": "start_session",
            "agent_id": "onecx-agent",
            "chat_id": "chat-1",
            "apm_principal_token": "t-1",
        }
    )
    assert ws.receive_json()["kind"] == "session_ready"


def _speak(ws) -> None:
    ws.send_json({"type": "user_started_speaking"})
    ws.send_json({"type": "audio", "audio": _AUDIO})
    ws.send_json({"type": "user_stopped_speaking"})


def test_a_failed_human_save_does_not_drop_the_next_turn():
    container = fake_container(token="t-1")
    persistence = FakePersistence(fail_human=True)
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _speak(ws)
        first = _until(ws, "error")
        assert first[-1]["code"] == "persistence_failed"
        persistence.fail_human = False
        _speak(ws)
        events = _until(ws, "turn_complete")
        assert [e["kind"] for e in events].count("turn_complete") == 1

        ws.send_json({"type": "stop"})
        _until(ws, "session_closed")

    assert [e.kind for e in persistence.entries] == [
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
    ]


def test_barge_in_after_a_slow_human_save_persists_the_interrupted_turn_first(slow_tts):
    container = fake_container(token="t-1")
    persistence = FakePersistence(human_delay=0.3)
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _speak(ws)
        _until(ws, "assistant_text")
        ws.send_json({"type": "user_started_speaking"})
        events = _until(ws, "turn_complete")
        assert "interrupted" in [e["kind"] for e in events]

        ws.send_json({"type": "stop"})
        _until(ws, "session_closed")

    assert [(e.kind, e.status) for e in persistence.entries] == [
        (VoiceEntryKind.HUMAN, VoiceEntryStatus.COMPLETED),
        (VoiceEntryKind.ASSISTANT, VoiceEntryStatus.INTERRUPTED),
    ]
