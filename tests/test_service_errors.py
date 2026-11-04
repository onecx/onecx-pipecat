import asyncio
import base64
import threading

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
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
from onecx_pipecat.pipeline import persistence as persistence_module
from onecx_pipecat.services.inmemory import InMemoryLLM, InMemorySTT, InMemoryTTS
from tests.fakes import FakePersistence, fake_container
from tests.voice_helpers import (
    GatedTTS,
    _receive_json_bounded,
    _speak,
    _until_bounded,
    end_session,
    wait_for_close,
)

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
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})
        events = []
        while not events or events[-1]["kind"] != "error":
            events.append(_receive_json_bounded(ws))
        assert events[-1]["code"] == "llm_failed"

        end_session(ws)


def test_stt_failure_is_reported_without_closing_and_next_turn_works(monkeypatch):
    class FailingSTT(InMemorySTT):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = True

        async def process_frame(self, frame, direction):
            if isinstance(frame, InputAudioRawFrame) and self.fail_next:
                self.fail_next = False
                await self.push_error("synthetic STT failure")
            await super().process_frame(frame, direction)

    stt = FailingSTT()
    monkeypatch.setattr(session_module, "build_stt_service", lambda *args: stt)
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
        _receive_json_bounded(ws)
        _speak(ws)
        first = _until_bounded(ws, "turn_complete")
        assert any(event.get("code") == "stt_failed" for event in first)
        assert not any(event["kind"] == "session_closed" for event in first)
        _speak(ws)
        second = _until_bounded(ws, "turn_complete")
        assert any(event["kind"] == "assistant_text" for event in second)
        end_session(ws)


def test_assistant_save_failure_has_no_completion_and_next_turn_works():
    persistence = FakePersistence(fail_assistant=True)
    container = fake_container(token="t-1")
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
        _receive_json_bounded(ws)
        _speak(ws)
        first = _until_bounded(ws, "error")
        assert first[-1]["code"] == "persistence_failed"
        assert not any(event["kind"] == "turn_complete" for event in first)
        assert not any(event["kind"] == "session_closed" for event in first)

        persistence.fail_assistant = False
        _speak(ws)
        second = _until_bounded(ws, "turn_complete")
        assert any(event["kind"] == "assistant_text" for event in second)
        end_session(ws)


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
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
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
        end_session(ws)

    assistant = [entry for entry in persistence.entries if entry.kind is VoiceEntryKind.ASSISTANT]
    assert [entry.text for entry in assistant] == ["normal answer"]


def test_context_reconciliation_failure_reports_llm_error_and_closes(monkeypatch):
    class CorruptingLLM(InMemoryLLM):
        def __init__(self) -> None:
            super().__init__(reply="visible")
            self.inferences = 0

        async def process_frame(self, frame, direction):
            if isinstance(frame, LLMContextFrame):
                self.inferences += 1
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

    llm = CorruptingLLM()
    monkeypatch.setattr(session_module, "build_llm", lambda *args: llm)
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
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})
        events = _until_bounded(ws, "assistant_text")
        caption = next(event["text"] for event in events if event["kind"] == "assistant_text")
        ws.send_json({"type": "user_started_speaking"})
        events += _until_bounded(ws, "session_closed")
        closed = wait_for_close(ws)

    final_events = [
        (event["kind"], event.get("code"), event.get("reason"))
        for event in events
        if event["kind"] in {"error", "session_closed"}
    ]
    assert final_events == [
        ("error", "llm_failed", None),
        ("session_closed", None, "context_inconsistent"),
    ]
    assert closed["code"] == 1000
    assert llm.inferences == 1
    assistant = [entry for entry in persistence.entries if entry.kind is VoiceEntryKind.ASSISTANT]
    assert [(entry.status.value, entry.text) for entry in assistant] == [("INTERRUPTED", caption)]


def test_stop_reason_is_preserved_when_context_reconciliation_fails(monkeypatch):
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
        _receive_json_bounded(ws)
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})
        _until_bounded(ws, "assistant_text")
        ws.send_json({"type": "stop"})
        events = _until_bounded(ws, "session_closed")
        closed = wait_for_close(ws)

    assert events[-1]["reason"] == "client_stopped"
    assert closed["code"] == 1000


def test_stop_closes_after_a_hung_assistant_save_times_out(monkeypatch):
    monkeypatch.setattr(persistence_module, "_TEARDOWN_SAVE_TIMEOUT_SECS", 0.01)
    entered = threading.Event()

    class HungPersistence(FakePersistence):
        async def append(self, session_id, entry, **kwargs):
            if entry.kind is VoiceEntryKind.ASSISTANT:
                entered.set()
                await asyncio.Future()
            return await super().append(session_id, entry, **kwargs)

    class LongTTS(InMemoryTTS):
        async def run_tts(self, text: str, context_id: str):
            yield TTSAudioRawFrame(
                b"\x00\x00" * self.sample_rate * 5,
                self.sample_rate,
                1,
                context_id=context_id,
            )

    container = fake_container(token="t-1")
    container.persistence = HungPersistence()  # type: ignore[assignment]
    monkeypatch.setattr(session_module, "build_llm", lambda *args: InMemoryLLM(reply="visible"))
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: LongTTS(sample_rate=settings.speech.sample_rate_hz),
    )
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
        _receive_json_bounded(ws)
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})
        _until_bounded(ws, "assistant_text")
        ws.send_json({"type": "stop"})
        assert entered.wait(timeout=3)
        closed = _until_bounded(ws, "session_closed")[-1]
        close_message = wait_for_close(ws)
    assert closed["reason"] == "client_stopped"
    assert close_message["code"] == 1000


def test_token_expiry_closes_while_audio_is_playing(monkeypatch):
    tts = GatedTTS(sample_rate=16000, wait_after_audio=True, audio_seconds=10)
    monkeypatch.setattr(session_module, "build_tts_service", lambda *args: tts)
    container = fake_container(token="t-1")
    client = TestClient(create_app(container, expiry_check_seconds=0.01))
    with client.websocket_connect("/ws/voice") as ws:
        try:
            ws.send_json(
                {
                    "type": "start_session",
                    "agent_id": "onecx-agent",
                    "chat_id": "chat-1",
                    "apm_principal_token": "t-1",
                }
            )
            _receive_json_bounded(ws)
            _speak(ws)
            assert tts.entered.wait(timeout=3)
            _until_bounded(ws, "assistant_audio")
            container.auth.mark_expired("t-1")
            events = _until_bounded(ws, "session_closed")
            close_message = wait_for_close(ws)
            assert any(event.get("code") == "token_expired" for event in events)
            assert events[-1]["reason"] == "token_expired"
            assert close_message["code"] == 1008
        finally:
            tts.release.set()


@pytest.mark.parametrize("audio_before_error", [False, True])
def test_tts_failure_reports_error_and_keeps_emitted_text(monkeypatch, audio_before_error):
    class FailingTTS(InMemoryTTS):
        def __init__(self, sample_rate: int) -> None:
            super().__init__(sample_rate=sample_rate)
            self.fail = True

        async def run_tts(self, text: str, context_id: str):
            if self.fail:
                if audio_before_error:
                    yield TTSAudioRawFrame(
                        b"\x00\x00" * 1600, self.sample_rate, 1, context_id=context_id
                    )
                yield ErrorFrame(error="synthetic TTS failure")
                return
            async for frame in super().run_tts(text, context_id):
                yield frame

    tts = FailingTTS(sample_rate=16000)
    monkeypatch.setattr(
        session_module,
        "build_tts_service",
        lambda settings, language: tts,
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
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})
        events = []
        while not (
            any(e["kind"] == "error" and e["code"] == "tts_failed" for e in events)
            and (not audio_before_error or any(e["kind"] == "turn_complete" for e in events))
        ):
            events.append(_receive_json_bounded(ws))
        tts.fail = False
        ws.send_json({"type": "user_started_speaking"})
        ws.send_json({"type": "audio", "audio": _AUDIO})
        ws.send_json({"type": "user_stopped_speaking"})
        follow_up = _until_bounded(ws, "assistant_text")
        follow_up += _until_bounded(ws, "turn_complete")
        end_session(ws)

    captions = [
        event["text"] for event in [*events, *follow_up] if event["kind"] == "assistant_text"
    ]
    assistant = [e for e in persistence.entries if e.kind is VoiceEntryKind.ASSISTANT]
    assert len(captions) >= 1
    assert [entry.text for entry in assistant] == captions
    assert [entry.status.value for entry in assistant] == (
        ["COMPLETED", "COMPLETED"] if audio_before_error else ["INTERRUPTED", "COMPLETED"]
    )
