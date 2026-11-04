"""Shared deterministic helpers for voice WebSocket tests."""

from __future__ import annotations

import asyncio
import base64
import json
import threading

import anyio
from pipecat.frames.frames import TTSAudioRawFrame

from onecx_pipecat.services.inmemory import InMemoryTTS

_AUDIO = base64.b64encode(b"\x01\x00" * 160).decode("ascii")


def _receive_raw_bounded(ws, timeout: float = 3) -> dict:
    async def receive() -> dict:
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()

    return ws.portal.call(receive)


def _receive_json_bounded(ws, timeout: float = 3) -> dict:
    message = _receive_raw_bounded(ws, timeout)
    ws._raise_on_close(message)
    return json.loads(message["text"])


def _until(ws, kind: str, timeout: float = 3) -> list[dict]:
    events = []
    while True:
        events.append(_receive_json_bounded(ws, timeout))
        if events[-1]["kind"] == kind:
            return events


def _until_bounded(ws, kind: str, timeout: float = 3) -> list[dict]:
    return _until(ws, kind, timeout)


def _start(ws, token: str = "t-1") -> None:
    ws.send_json(
        {
            "type": "start_session",
            "agent_id": "onecx-agent",
            "chat_id": "chat-1",
            "apm_principal_token": token,
        }
    )
    assert _receive_json_bounded(ws)["kind"] == "session_ready"


def _speak(ws) -> None:
    ws.send_json({"type": "user_started_speaking"})
    ws.send_json({"type": "audio", "audio": _AUDIO})
    ws.send_json({"type": "user_stopped_speaking"})


def _wait_for_close(ws, timeout: float = 3) -> dict:
    while (message := _receive_raw_bounded(ws, timeout))["type"] != "websocket.close":
        continue
    return message


def wait_for_close(ws, timeout: float = 3) -> dict:
    return _wait_for_close(ws, timeout)


def end_session(ws) -> None:
    """Stop the session and drain its server close before TestClient exits.

    TestClient cancels the ASGI task on context exit. If that lands during
    Pipecat's bus shutdown, TaskManager can re-raise an inner CancelledError
    without anyio's cancel-scope marker and produce a flaky CancelledError.
    """
    ws.send_json({"type": "stop"})
    _until_bounded(ws, "session_closed")
    _wait_for_close(ws)


def disconnect_and_drain(ws) -> None:
    """Disconnect the client and drain the server-owned WebSocket close."""
    ws.close()
    _wait_for_close(ws)


class GatedTTS(InMemoryTTS):
    """TTS fake with synthesis gates and configurable audio playback duration."""

    def __init__(
        self,
        sample_rate: int,
        *,
        wait_before_audio: bool = False,
        wait_after_audio: bool = False,
        audio_seconds: float = 1,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.wait_before_audio = wait_before_audio
        self.wait_after_audio = wait_after_audio
        self.audio_seconds = audio_seconds

    async def run_tts(self, text: str, context_id: str):
        self.entered.set()
        if self.wait_before_audio:
            await asyncio.to_thread(self.release.wait)
        yield TTSAudioRawFrame(
            b"\x00\x00" * int(self.sample_rate * self.audio_seconds),
            self.sample_rate,
            1,
            context_id=context_id,
        )
        if self.wait_after_audio:
            await asyncio.to_thread(self.release.wait)
