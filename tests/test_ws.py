import base64

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from onecx_pipecat.app import create_app
from onecx_pipecat.config import Settings, VoiceSettings
from onecx_pipecat.container import AppContainer
from tests.fakes import FakePersistence, fake_container
from tests.voice_helpers import (
    _receive_json_bounded,
    _speak,
    _until,
    end_session,
    wait_for_close,
)

_AUDIO = base64.b64encode(b"\x01\x00" * 160).decode("ascii")


def _container() -> AppContainer:
    return fake_container(token="t-1")


def _start(token: str = "t-1") -> dict:
    return {
        "type": "start_session",
        "agent_id": "onecx-agent",
        "chat_id": "chat-1",
        "apm_principal_token": token,
    }


def test_ws_rejects_token_in_url():
    client = TestClient(create_app(_container()))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/voice?token=t-1") as ws:
            _receive_json_bounded(ws)


@pytest.mark.parametrize(
    "query", ["token=t-1", "apm-principal-token=t-1", "access_token=t-1", "authorization=x", "a=b"]
)
def test_ws_rejects_any_query_parameter(query):
    client = TestClient(create_app(_container()))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/voice?{query}") as ws:
            _receive_json_bounded(ws)


def test_ws_failed_start_sends_error_then_closes_with_policy_violation():
    client = TestClient(create_app(_container()))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start(token="nope"))
        assert _receive_json_bounded(ws)["code"] == "authentication_failed"
        assert _receive_json_bounded(ws)["kind"] == "session_closed"
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_json_bounded(ws)
        assert closed.value.code == 1008


def test_ws_malformed_handshake_is_reported_and_ignored():
    client = TestClient(create_app(_container()))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "nonsense"})
        assert _receive_json_bounded(ws)["code"] == "bad_command"
        ws.send_text("not json")
        assert _receive_json_bounded(ws)["code"] == "bad_command"
        ws.send_json({"type": "audio", "audio": _AUDIO})
        assert _receive_json_bounded(ws)["code"] == "bad_command"
        ws.send_json(_start())
        ready = _receive_json_bounded(ws)
        assert ready["kind"] == "session_ready"
        assert ready["schema_version"] == "1"
        end_session(ws)


def test_ws_voice_turn_streams_transcript_text_audio_and_completion():
    client = TestClient(create_app(_container()))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        _speak(ws)
        events = _until(ws, "turn_complete")
        kinds = {e["kind"] for e in events}
        assert {"transcript", "assistant_text", "assistant_audio", "turn_complete"} <= kinds
        final = [e for e in events if e["kind"] == "transcript" and e["state"] == "FINAL"]
        assert final[-1]["text"] == "in-memory transcript"
        assert "".join(e["text"] for e in events if e["kind"] == "assistant_text").strip()

        ws.send_json({"type": "stop"})
        assert _until(ws, "session_closed")[-1]["reason"] == "client_stopped"
        assert wait_for_close(ws)["code"] == 1000


def test_ws_turn_is_persisted():
    container = _container()
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        _receive_json_bounded(ws)
        _speak(ws)
        _until(ws, "turn_complete")
        end_session(ws)
    assert [e.kind.value for e in persistence.entries] == ["HUMAN", "ASSISTANT"]
    assert persistence.entries[0].text == "in-memory transcript"


def test_ws_malformed_message_while_running_is_reported_and_session_continues():
    client = TestClient(create_app(_container()))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        _receive_json_bounded(ws)
        ws.send_json({"type": "nonsense"})
        assert _until(ws, "error")[-1]["code"] == "bad_command"
        _speak(ws)
        _until(ws, "turn_complete")
        end_session(ws)


def test_ws_refresh_token_keeps_the_session_alive():
    container = _container()
    container.auth.seed("t-2", subject="user-1", tenant_id="local", roles=["voice"])
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        _receive_json_bounded(ws)
        ws.send_json({"type": "refresh_token", "apm_principal_token": "t-2"})
        _speak(ws)
        _until(ws, "turn_complete")
        end_session(ws)


def test_ws_refresh_with_a_bad_token_is_a_typed_error():
    client = TestClient(create_app(_container()))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        _receive_json_bounded(ws)
        ws.send_json({"type": "refresh_token", "apm_principal_token": "bogus"})
        assert _until(ws, "error")[-1]["code"] == "authentication_failed"
        end_session(ws)


def test_ws_idle_session_is_ended_at_token_expiry():
    container = _container()
    client = TestClient(create_app(container, expiry_check_seconds=0.05))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        container.auth.mark_expired("t-1")
        assert _until(ws, "error")[-1]["code"] == "token_expired"
        assert _receive_json_bounded(ws)["kind"] == "session_closed"
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_json_bounded(ws)
        assert closed.value.code == 1008


def _same_host_container() -> AppContainer:
    container = _container()
    container.settings = Settings(
        llm=container.settings.llm,
        speech=container.settings.speech,
        voice=VoiceSettings(origin_policy="same-host"),
    )
    return container


def _off_container() -> AppContainer:
    container = _container()
    container.settings = Settings(
        llm=container.settings.llm,
        speech=container.settings.speech,
        voice=VoiceSettings(origin_policy="off"),
    )
    return container


def test_ws_deny_policy_rejects_even_a_same_host_origin():
    # Default policy is ``deny``: every connection is rejected before a session
    # is created, regardless of the Origin header.
    container = _container()
    container.settings.voice.origin_policy = "deny"
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice", headers={"origin": "http://testserver"}) as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_json_bounded(ws)
        assert closed.value.code == 1008
        assert closed.value.reason == "origin_not_allowed"


def test_ws_same_host_allows_matching_origin():
    client = TestClient(create_app(_same_host_container()))
    with client.websocket_connect("/ws/voice", headers={"origin": "http://testserver"}) as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        end_session(ws)


def test_ws_same_host_allows_matching_origin_with_a_different_port():
    # The comparison is hostname-only: scheme and port are ignored.
    client = TestClient(create_app(_same_host_container()))
    with client.websocket_connect("/ws/voice", headers={"origin": "https://testserver:443"}) as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        end_session(ws)


def test_ws_same_host_rejects_a_foreign_origin():
    client = TestClient(create_app(_same_host_container()))
    with client.websocket_connect("/ws/voice", headers={"origin": "https://evil.example"}) as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_json_bounded(ws)
        assert closed.value.code == 1008
        assert closed.value.reason == "origin_not_allowed"


def test_ws_same_host_rejects_a_missing_origin():
    client = TestClient(create_app(_same_host_container()))
    with client.websocket_connect("/ws/voice") as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_json_bounded(ws)
        assert closed.value.code == 1008
        assert closed.value.reason == "origin_not_allowed"


def test_ws_same_host_rejects_a_null_origin():
    client = TestClient(create_app(_same_host_container()))
    with client.websocket_connect("/ws/voice", headers={"origin": "null"}) as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            _receive_json_bounded(ws)
        assert closed.value.code == 1008
        assert closed.value.reason == "origin_not_allowed"


def test_ws_off_policy_allows_any_origin():
    # ``off`` skips the check: a foreign origin still opens a session. Intended
    # for tests and local development only.
    client = TestClient(create_app(_off_container()))
    with client.websocket_connect("/ws/voice", headers={"origin": "https://evil.example"}) as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        end_session(ws)


def test_ws_off_policy_allows_a_missing_origin():
    client = TestClient(create_app(_off_container()))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        end_session(ws)
