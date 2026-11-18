import base64
import time

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


@pytest.mark.parametrize("reference", [None, "config-v1"])
def test_ws_turn_is_persisted_with_pinned_version(reference):
    container = _container()
    persistence = FakePersistence()
    container.persistence = persistence  # type: ignore[assignment]
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({**_start(), "agent_config_version": reference})
        assert _receive_json_bounded(ws)["agent_config_version"] == "config-v1"
        _speak(ws)
        _until(ws, "turn_complete")
        end_session(ws)
    assert [e.kind.value for e in persistence.entries] == ["HUMAN", "ASSISTANT"]
    assert persistence.entries[0].text == "in-memory transcript"
    assert all(entry.agent_config_version == "config-v1" for entry in persistence.entries)


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


def _expect_limit_end(ws, reason: str, close_code: int) -> None:
    assert _until(ws, "error")[-1]["code"] == reason
    assert _receive_json_bounded(ws) == {"kind": "session_closed", "reason": reason}
    with pytest.raises(WebSocketDisconnect) as closed:
        _receive_json_bounded(ws)
    assert closed.value.code == close_code


def test_ws_session_ends_at_the_maximum_duration():
    container = _container()
    container.settings.pipeline.max_session_duration_secs = 0.2
    client = TestClient(create_app(container, expiry_check_seconds=0.05))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        _expect_limit_end(ws, "session_max_duration", 1000)


def test_ws_idle_session_ends_with_a_typed_idle_timeout():
    container = _container()
    container.settings.pipeline.idle_timeout_secs = 0.3
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        _expect_limit_end(ws, "session_idle_timeout", 1000)


def test_ws_a_push_to_talk_turn_counts_as_activity_for_the_idle_timeout():
    # Client turn detection has no VAD, so no UserSpeakingFrame ever arrives; the
    # turn start is the only user activity. Starting a turn halfway through the
    # idle timeout must therefore push the idle end out by about half the timeout.
    idle = 0.8
    container = _container()
    container.settings.pipeline.idle_timeout_secs = idle
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        ready = time.monotonic()
        time.sleep(0.5)
        ws.send_json({"type": "user_started_speaking"})
        events = _until(ws, "session_closed")
        elapsed = time.monotonic() - ready
        assert events[-1] == {"kind": "session_closed", "reason": "session_idle_timeout"}
        assert elapsed >= 0.5 + idle - 0.2, f"idle fired {elapsed:.2f}s after ready"
        wait_for_close(ws)


def test_ws_oversized_audio_ends_the_session_with_a_typed_outcome():
    container = _container()
    container.settings.pipeline.max_audio_message_bytes = 1280
    client = TestClient(create_app(container))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        ws.send_json({"type": "audio", "audio": _AUDIO})  # 320 bytes: accepted
        ws.send_json({"type": "audio", "audio": base64.b64encode(b"\x01" * 1281).decode("ascii")})
        _expect_limit_end(ws, "audio_message_too_large", 1008)


def test_ws_limits_do_not_fire_for_a_session_within_them():
    container = _container()
    container.settings.pipeline.max_audio_message_bytes = 1280
    container.settings.pipeline.idle_timeout_secs = 5
    container.settings.pipeline.max_session_duration_secs = 60
    client = TestClient(create_app(container, expiry_check_seconds=0.05))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_start())
        assert _receive_json_bounded(ws)["kind"] == "session_ready"
        _speak(ws)
        time.sleep(0.3)
        ws.send_json({"type": "stop"})
        assert _until(ws, "session_closed")[-1]["reason"] == "client_stopped"
        wait_for_close(ws)
