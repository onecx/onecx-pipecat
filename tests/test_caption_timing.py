import asyncio
import threading

from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator

from onecx_pipecat import session as session_module
from onecx_pipecat.app import create_app
from onecx_pipecat.models import VoiceEntryKind
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


def test_each_caption_reaches_the_client_before_its_audio_has_played(monkeypatch):
    tts = GatedTTS(sample_rate=16000, wait_after_audio=True, audio_seconds=10)
    monkeypatch.setattr(
        session_module,
        "build_llm",
        lambda *args: InMemoryLLM(reply="Sentence 0. Sentence 1. Sentence 2."),
    )
    monkeypatch.setattr(session_module, "build_tts_service", lambda *args: tts)
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            assert tts.entered.wait(timeout=3)
            events = []
            while not any(event["kind"] == "assistant_audio" for event in events):
                events.append(_receive_json_bounded(ws))
            tts.release.set()
            while sum(event["kind"] == "assistant_text" for event in events) < 3:
                events.append(_receive_json_bounded(ws))
            captions = [event["text"] for event in events if event["kind"] == "assistant_text"]
            assert captions == ["Sentence 0.", "Sentence 1.", "Sentence 2."]
            assert not any(event["kind"] == "bot_stopped_speaking" for event in events)
            end_session(ws)
        finally:
            tts.release.set()


def test_caption_waits_for_synthesis_to_finish(monkeypatch):
    tts = GatedTTS(sample_rate=16000, wait_after_audio=True)
    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="One sentence.")
    )
    monkeypatch.setattr(session_module, "build_tts_service", lambda *args: tts)
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            assert tts.entered.wait(timeout=3)
            events = []
            while True:
                event = _receive_json_bounded(ws)
                events.append(event)
                if event["kind"] == "assistant_audio":
                    break
            assert event["kind"] == "assistant_audio"
            assert not any(e["kind"] == "assistant_text" for e in events)
            tts.release.set()
            events += _until(ws, "assistant_text")
            assert events[-1]["text"] == "One sentence."
            end_session(ws)
        finally:
            tts.release.set()


def test_barge_in_during_synthesis_does_not_publish_a_caption(monkeypatch):
    tts = GatedTTS(sample_rate=16000, wait_before_audio=True)
    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="Visible before audio.")
    )
    monkeypatch.setattr(session_module, "build_tts_service", lambda *args: tts)
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            assert tts.entered.wait(timeout=3)
            ws.send_json({"type": "user_started_speaking"})
            events = _until(ws, "interrupted")
            assert not any(e["kind"] == "assistant_audio" for e in events)
            assert not any(e["kind"] == "assistant_text" for e in events)
            tts.release.set()
            end_session(ws)
        finally:
            tts.release.set()

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert assistant == []


def test_barge_in_during_synthesis_before_aggregator_start_has_no_caption(monkeypatch):
    release_start = threading.Event()
    start_waiting = threading.Event()
    persisted = threading.Event()
    original_process = LLMAssistantAggregator.process_frame
    tts = GatedTTS(sample_rate=16000, wait_before_audio=True)

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

    monkeypatch.setattr(LLMAssistantAggregator, "process_frame", delayed_start)
    monkeypatch.setattr(
        session_module, "build_llm", lambda *args: InMemoryLLM(reply="Caption before start.")
    )
    monkeypatch.setattr(session_module, "build_tts_service", lambda *args: tts)
    container = fake_container(token="t-1")
    persistence = SignalingPersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            assert tts.entered.wait(timeout=3)
            assert start_waiting.wait(timeout=3)
            ws.send_json({"type": "user_started_speaking"})
            events = _until(ws, "interrupted")
            assert not any(e["kind"] == "assistant_text" for e in events)
        finally:
            release_start.set()
            tts.release.set()
        end_session(ws)

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert assistant == []
