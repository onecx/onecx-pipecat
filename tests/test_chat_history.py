from fastapi.testclient import TestClient
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection

from onecx_pipecat import session as session_module
from onecx_pipecat.app import create_app
from onecx_pipecat.protocol import StartSession
from onecx_pipecat.services.inmemory import InMemoryLLM
from onecx_pipecat.session import SessionState, VoiceSession
from tests.fakes import FakePersistence, fake_container
from tests.voice_helpers import _receive_json_bounded, _speak, _until_bounded, end_session

_HISTORY = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer"},
]


class RecordingLLM(InMemoryLLM):
    seen: list[list[dict]] = []

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMContextFrame):
            self.seen.append([dict(m) for m in frame.context.get_messages()])
        await super().process_frame(frame, direction)


def _start_message() -> dict:
    return {
        "type": "start_session",
        "agent_id": "onecx-agent",
        "chat_id": "chat-1",
        "apm_principal_token": "t-1",
    }


def _start_command() -> StartSession:
    return StartSession(agent_id="onecx-agent", chat_id="chat-1", apm_principal_token="tok-user")


def test_chat_history_seeds_the_llm_context_and_turns_are_saved_to_the_chat(monkeypatch):
    RecordingLLM.seen = []
    monkeypatch.setattr(session_module, "build_llm", lambda *args: RecordingLLM())
    container = fake_container(token="t-1")
    persistence = FakePersistence()
    persistence.history_messages = list(_HISTORY)
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start_message())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        _speak(ws)
        _until_bounded(ws, "turn_complete")
        end_session(ws)
    messages = RecordingLLM.seen[0]
    assert [m["role"] for m in messages[:3]] == ["system", "user", "assistant"]
    assert [m["content"] for m in messages[1:3]] == ["earlier question", "earlier answer"]
    assert messages[-1] == {"role": "user", "content": "in-memory transcript"}
    assert persistence.appended_with == [("chat-1", "t-1", "user-1")] * 2


async def test_history_is_limited_by_the_configured_maximum():
    container = fake_container()
    container.settings.chat.history_max_messages = 1
    container.persistence.history_messages = list(_HISTORY)  # type: ignore[attr-defined]
    session = VoiceSession("s", container)
    await session.dispatch(_start_command())
    assert session._history == _HISTORY[-1:]


async def test_history_load_failure_fails_the_start_closed():
    container = fake_container()
    container.persistence.fail_history = True  # type: ignore[attr-defined]
    session = VoiceSession("s", container)
    events = await session.dispatch(_start_command())
    assert [e.kind for e in events] == ["error", "session_closed"]
    assert events[0].code == "chat_not_accessible"
    assert session.state is SessionState.CLOSED
