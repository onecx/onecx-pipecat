import asyncio
import threading

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import LLMFullResponseEndFrame
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator

from onecx_pipecat import session as session_module
from onecx_pipecat.app import create_app
from onecx_pipecat.models import VoiceEntryKind, VoiceEntryStatus
from onecx_pipecat.services.inmemory import InMemoryLLM, InMemoryTTS
from tests.fakes import FakePersistence, fake_container
from tests.voice_helpers import (
    _AUDIO,
    GatedTTS,
    _receive_json_bounded,
    _speak,
    _start,
    _until,
    end_session,
    wait_for_close,
)

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
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})
        _until(ws, "assistant_text")

        ws.send_json({"type": "user_started_speaking"})
        events = _until(ws, "turn_complete")
        assert "interrupted" in [e["kind"] for e in events]

        end_session(ws)

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [e.status for e in assistant] == [VoiceEntryStatus.INTERRUPTED]


@pytest.mark.parametrize("ending", ["stop", "disconnect"])
def test_ending_session_persists_caption_before_audio(monkeypatch, ending):
    entered = threading.Event()
    release = threading.Event()
    persisted = threading.Event()

    class SlowPersistence(FakePersistence):
        async def append(self, session_id, entry, **kwargs):
            if entry.kind is VoiceEntryKind.ASSISTANT:
                entered.set()
                await asyncio.to_thread(release.wait)
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
        lambda settings, language: GatedTTS(
            sample_rate=settings.speech.sample_rate_hz, audio_seconds=1
        ),
    )
    container = fake_container(token="t-1")
    persistence = SlowPersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _speak(ws)
        _until(ws, "assistant_text")
        if ending == "stop":
            ws.send_json({"type": "stop"})
            assert entered.wait(timeout=3)
            release.set()
            _until(ws, "session_closed")
            wait_for_close(ws)
        else:
            ws.close()
            assert entered.wait(timeout=3)
            release.set()
            wait_for_close(ws)
            assert persisted.wait(timeout=3)
        release.set()

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.INTERRUPTED, "Caption before stop.")
    ]


def test_barge_in_persists_every_caption_shown_not_only_the_spoken_ones(monkeypatch):
    tts = GatedTTS(sample_rate=16000, wait_after_audio=True, audio_seconds=10)
    monkeypatch.setattr(
        session_module,
        "build_llm",
        lambda *args: InMemoryLLM(reply="Sentence 0. Sentence 1. Sentence 2."),
    )
    monkeypatch.setattr(session_module, "build_tts_service", lambda *args: tts)
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
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
        ws.send_json({"type": "user_started_speaking"})
        events += _until(ws, "turn_complete")
        assert "interrupted" in [e["kind"] for e in events]

        end_session(ws)

    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert [(e.status, e.text) for e in assistant] == [
        (VoiceEntryStatus.INTERRUPTED, " ".join(captions))
    ]


def test_caption_is_persisted_once_if_interruption_precedes_response_end(monkeypatch):
    end_waiting = threading.Event()
    release_end = threading.Event()
    original_process = LLMAssistantAggregator.process_frame

    async def delayed_end(self, frame, direction):
        if isinstance(frame, LLMFullResponseEndFrame):
            end_waiting.set()
            await asyncio.to_thread(release_end.wait)
        await original_process(self, frame, direction)

    monkeypatch.setattr(LLMAssistantAggregator, "process_frame", delayed_end)
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            _start(ws)
            _speak(ws)
            assert end_waiting.wait(timeout=3)
            events = _until(ws, "assistant_text")
            caption = next(event["text"] for event in events if event["kind"] == "assistant_text")
            ws.send_json({"type": "user_started_speaking"})
            events += _until(ws, "turn_complete")
            assert "interrupted" in [event["kind"] for event in events]
        finally:
            release_end.set()
        end_session(ws)

    assistant = [entry for entry in persistence.entries if entry.kind is VoiceEntryKind.ASSISTANT]
    assert [(entry.status, entry.text) for entry in assistant] == [
        (VoiceEntryStatus.INTERRUPTED, caption)
    ]


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

        end_session(ws)

    assert [e.kind for e in persistence.entries] == [
        VoiceEntryKind.HUMAN,
        VoiceEntryKind.ASSISTANT,
    ]


def test_barge_in_after_a_slow_human_save_persists_the_interrupted_turn_first(slow_tts):
    container = fake_container(token="t-1")

    entered = threading.Event()
    release = threading.Event()

    class GatedPersistence(FakePersistence):
        async def append(self, session_id, entry, **kwargs):
            if entry.kind is VoiceEntryKind.HUMAN:
                entered.set()
                await asyncio.to_thread(release.wait)
            return await super().append(session_id, entry, **kwargs)

    persistence = GatedPersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _speak(ws)
        _until(ws, "assistant_text")
        assert entered.wait(timeout=3)
        ws.send_json({"type": "user_started_speaking"})
        release.set()
        events = _until(ws, "turn_complete")
        assert "interrupted" in [e["kind"] for e in events]
        end_session(ws)

    assert [(e.kind, e.status) for e in persistence.entries] == [
        (VoiceEntryKind.HUMAN, VoiceEntryStatus.COMPLETED),
        (VoiceEntryKind.ASSISTANT, VoiceEntryStatus.INTERRUPTED),
    ]
