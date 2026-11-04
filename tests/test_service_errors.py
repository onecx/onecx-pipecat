import base64

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import ErrorFrame, LLMContextFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

from onecx_pipecat import session as session_module
from onecx_pipecat.app import create_app
from onecx_pipecat.models import VoiceEntryKind
from onecx_pipecat.services.inmemory import InMemoryLLM, InMemoryTTS
from tests.fakes import FakePersistence, fake_container

_AUDIO = base64.b64encode(b"\x01\x00" * 160).decode("ascii")


class FailingLLM(InMemoryLLM):
    async def process_frame(self, frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMContextFrame):
            await self.push_error("boom")
            return
        await super().process_frame(frame, direction)


def test_llm_failure_is_a_typed_non_fatal_error(monkeypatch):
    monkeypatch.setattr(session_module, "build_llm", lambda *args: FailingLLM())
    client = TestClient(create_app(fake_container(token="t-1")))
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
        events = []
        while not events or events[-1]["kind"] != "error":
            events.append(ws.receive_json())
        assert events[-1]["code"] == "llm_failed"

        ws.send_json({"type": "stop"})
        while ws.receive_json()["kind"] != "session_closed":
            pass


@pytest.mark.parametrize("audio_before_error", [False, True])
def test_tts_failure_reports_error_and_keeps_emitted_text(monkeypatch, audio_before_error):
    class FailingTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            if audio_before_error:
                yield TTSAudioRawFrame(
                    b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id
                )
            yield ErrorFrame(error="synthetic TTS failure")

    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: FailingTTS(sample_rate=settings.speech.sample_rate_hz),
    )
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
        events = []
        while not (
            any(e["kind"] == "turn_complete" for e in events)
            and any(e["kind"] == "error" and e["code"] == "tts_failed" for e in events)
        ):
            events.append(ws.receive_json())
        ws.send_json({"type": "stop"})
        while ws.receive_json()["kind"] != "session_closed":
            pass

    captions = [e["text"] for e in events if e["kind"] == "assistant_text"]
    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert len(captions) == 1
    assert [entry.text for entry in assistant] == captions
