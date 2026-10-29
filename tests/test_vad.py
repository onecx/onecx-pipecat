"""Server-side turn detection: the client only streams audio, the VAD finds the turns."""

import base64

import pytest
from fastapi.testclient import TestClient

from onecx_pipecat import session as session_module
from onecx_pipecat.app import create_app
from onecx_pipecat.config import Settings, SpeechSettings
from onecx_pipecat.services.inmemory import InMemoryLLM, InMemoryTTS
from onecx_pipecat.services.speech import build_vad_analyzer
from tests.fakes import FakeVAD, fake_container

_VOICE = base64.b64encode(b"\x01\x10" * 160).decode("ascii")
_SILENCE = base64.b64encode(b"\x00\x00" * 160).decode("ascii")


@pytest.fixture
def vad(monkeypatch):
    monkeypatch.setattr(session_module, "build_vad_analyzer", lambda settings: FakeVAD())


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


def _say(ws, *, voiced: int = 10, silent: int = 20) -> None:
    for _ in range(voiced):
        ws.send_json({"type": "audio", "audio": _VOICE})
    for _ in range(silent):
        ws.send_json({"type": "audio", "audio": _SILENCE})


def _until(ws, kind: str) -> list[dict]:
    events = []
    while True:
        events.append(ws.receive_json())
        if events[-1]["kind"] == kind:
            return events


def test_speech_then_a_pause_completes_a_turn_without_client_turn_messages(vad):
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _say(ws)
        events = _until(ws, "turn_complete")
        kinds = [e["kind"] for e in events]
        assert kinds.index("user_started_speaking") < kinds.index("user_stopped_speaking")
        assert kinds.index("user_stopped_speaking") < kinds.index("assistant_text")
        final = [e for e in events if e["kind"] == "transcript" and e["state"] == "FINAL"]
        assert final[-1]["text"] == "in-memory transcript"

        ws.send_json({"type": "stop"})
        _until(ws, "session_closed")


def test_silence_alone_never_opens_a_turn(vad):
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _say(ws, voiced=0, silent=30)
        ws.send_json({"type": "stop"})
        kinds = [e["kind"] for e in _until(ws, "session_closed")]
        assert "user_started_speaking" not in kinds
        assert "turn_complete" not in kinds


def test_speaking_over_the_bot_interrupts_it(vad, monkeypatch):
    monkeypatch.setattr(
        session_module,
        "build_llm",
        lambda *args: InMemoryLLM(reply=" ".join(f"Sentence {i}." for i in range(30))),
    )
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: InMemoryTTS(
            sample_rate=settings.speech.sample_rate_hz, chunks=3, chunk_ms=100
        ),
    )
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        _say(ws)
        _until(ws, "assistant_text")

        _say(ws)
        kinds = [e["kind"] for e in _until(ws, "interrupted")]
        assert "user_started_speaking" in kinds

        ws.send_json({"type": "stop"})
        _until(ws, "session_closed")


def test_client_turn_detection_builds_no_vad():
    assert build_vad_analyzer(Settings(speech=SpeechSettings(turn_detection="client"))) is None
