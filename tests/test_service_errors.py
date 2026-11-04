import base64
import json
import threading

import anyio
import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    ErrorFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    TTSAudioRawFrame,
)
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
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_error("boom")
            await self.push_frame(LLMFullResponseEndFrame())
            return
        await super().process_frame(frame, direction)


class BoundaryThenNormalLLM(InMemoryLLM):
    def __init__(self, boundary: str) -> None:
        super().__init__(reply="normal answer")
        self._boundary = boundary
        self._first = True
        self.boundary_complete = threading.Event()

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMContextFrame) and self._first:
            self._first = False
            boundary = self._boundary
            await self.push_frame(LLMFullResponseStartFrame())
            if boundary == "error":
                await self.push_error("boom")
            elif boundary == "tool":
                await self.push_frame(
                    LLMMessagesAppendFrame(
                        messages=[
                            {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
                            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                        ],
                        run_llm=False,
                    )
                )
            await self.push_frame(LLMFullResponseEndFrame())
            self.boundary_complete.set()
            return
        await super().process_frame(frame, direction)


def _receive_json_bounded(ws, timeout: float = 3) -> dict:
    async def receive() -> dict:
        with anyio.fail_after(timeout):
            message = await ws._send_rx.receive()
        ws._raise_on_close(message)
        return json.loads(message["text"])

    return ws.portal.call(receive)


def _until_bounded(ws, kind: str) -> list[dict]:
    events = []
    while True:
        events.append(_receive_json_bounded(ws))
        if events[-1]["kind"] == kind:
            return events


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


@pytest.mark.parametrize("boundary", ["empty", "error", "tool"])
def test_empty_response_boundary_does_not_block_the_next_turn_or_stop(monkeypatch, boundary):
    llms = []

    def build_llm(*args):
        llm = BoundaryThenNormalLLM(boundary)
        llms.append(llm)
        return llm

    monkeypatch.setattr(session_module, "build_llm", build_llm)
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
        assert llms[0].boundary_complete.wait(timeout=3)
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})

        events = _until_bounded(ws, "turn_complete")
        assert sum(event["kind"] == "turn_complete" for event in events) == 1
        assert [event["code"] for event in events if event["kind"] == "error"] == (
            ["llm_failed"] if boundary == "error" else []
        )
        ws.send_json({"type": "stop"})
        assert _until_bounded(ws, "session_closed")[-1]["reason"] == "client_stopped"

    assistant = [entry for entry in persistence.entries if entry.kind is VoiceEntryKind.ASSISTANT]
    assert [entry.text for entry in assistant] == ["normal answer"]


def test_context_reconciliation_failure_reports_llm_error_and_closes(monkeypatch):
    class CorruptingLLM(InMemoryLLM):
        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                await super().process_frame(frame, direction)
                frame.context.set_messages(
                    [
                        message
                        for message in frame.context.get_messages()
                        if message.get("role") != "user"
                    ]
                )
                return
            await super().process_frame(frame, direction)

    class LongTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            yield TTSAudioRawFrame(
                b"\x00\x00" * self.sample_rate * 5,
                self.sample_rate,
                1,
                context_id=context_id,
            )

    monkeypatch.setattr(session_module, "build_llm", lambda *args: CorruptingLLM(reply="visible"))
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: LongTTS(sample_rate=settings.speech.sample_rate_hz),
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
        events = _until_bounded(ws, "assistant_text")
        caption = next(event["text"] for event in events if event["kind"] == "assistant_text")
        ws.send_json({"type": "user_started_speaking"})
        events += _until_bounded(ws, "session_closed")

    assert any(event["kind"] == "error" and event["code"] == "llm_failed" for event in events)
    assistant = [entry for entry in persistence.entries if entry.kind is VoiceEntryKind.ASSISTANT]
    assert [(entry.status.value, entry.text) for entry in assistant] == [("INTERRUPTED", caption)]


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
