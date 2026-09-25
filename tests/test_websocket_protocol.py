"""Integration tests for the authenticated FastAPI WebSocket voice-session transport."""

import pytest
from starlette.websockets import WebSocketDisconnect

from .conftest import AUTHENTICATED_PATH

FULL_TURN_PAYLOADS = [
    {"type": "START_SESSION", "session_type": "AI_CHAT"},
    {"type": "START_LISTENING"},
    {"type": "AUDIO_CHUNK", "audio_base64": "AAA=", "sequence": 1},
    {"type": "STOP_LISTENING"},
    {"type": "STOP_SESSION"},
]

EXPECTED_TURN_EVENT_TYPES = [
    "SESSION_READY",
    "LISTENING_STARTED",
    "LISTENING_STOPPED",
    "PROCESSING_STARTED",
    "TRANSCRIPT_PARTIAL",
    "TRANSCRIPT_FINAL",
    "ASSISTANT_TEXT",
    "ASSISTANT_TEXT",
    "ASSISTANT_TEXT",
    "ASSISTANT_AUDIO_CHUNK",
    "ASSISTANT_AUDIO_CHUNK",
    "SESSION_STOPPED",
]


def test_missing_auth_query_params_closes_connection(client):
    # The server rejects the handshake before accepting, so the close frame
    # (code 4401) arrives while the client is connecting. Starlette's TestClient
    # surfaces that as a WebSocketDisconnect raised from websocket_connect().
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/voice-sessions/AI_CHAT"):
            pass
    assert excinfo.value.code == 4401


def test_wrong_session_type_closes_connection(client):
    path = "/ws/voice-sessions/UNKNOWN" + "?tenant_id=tenant-a" + "&principal_id=user-1"
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(path):
            pass
    assert excinfo.value.code == 4400


def test_full_authenticated_turn_over_websocket(client):
    with client.websocket_connect(AUTHENTICATED_PATH) as websocket:
        for payload in FULL_TURN_PAYLOADS:
            websocket.send_json(payload)
        received = [websocket.receive_json() for _ in range(12)]
    assert [event["type"] for event in received] == EXPECTED_TURN_EVENT_TYPES


def test_server_stops_serving_after_session_stopped(client):
    # After the session stops (SESSION_STOPPED) the server must end the
    # connection rather than keep processing further commands on the same
    # socket. One WebSocket connection serves one voice session.
    with client.websocket_connect(AUTHENTICATED_PATH) as websocket:
        for payload in FULL_TURN_PAYLOADS:
            websocket.send_json(payload)
        received = [websocket.receive_json() for _ in range(12)]
        assert [event["type"] for event in received] == EXPECTED_TURN_EVENT_TYPES
        # The server has closed the socket: a further receive surfaces a
        # disconnect rather than another event.
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()


def test_illegal_command_over_websocket_yields_typed_error(client):
    with client.websocket_connect(AUTHENTICATED_PATH) as websocket:
        websocket.send_json({"type": "START_SESSION", "session_type": "AI_CHAT"})
        websocket.receive_json()
        websocket.send_json({"type": "AUDIO_CHUNK", "audio_base64": "AAA=", "sequence": 1})
        event = websocket.receive_json()
    assert event["type"] == "ERROR"
    assert event["code"] == "ILLEGAL_MICROPHONE_TRANSITION"


def test_degraded_capability_over_websocket(degraded_client):
    # Send START_SESSION and START_LISTENING, consuming their events.
    with degraded_client.websocket_connect(AUTHENTICATED_PATH) as websocket:
        websocket.send_json(FULL_TURN_PAYLOADS[0])
        websocket.receive_json()
        websocket.send_json(FULL_TURN_PAYLOADS[1])
        websocket.receive_json()
        # AUDIO_CHUNK emits no event (it only buffers audio); send it without
        # receiving anything.
        websocket.send_json(FULL_TURN_PAYLOADS[2])
        # STOP_LISTENING runs the turn: transcript + assistant + tts, and the
        # degraded assistant adapter fails on the first response chunk.
        websocket.send_json(FULL_TURN_PAYLOADS[3])
        events = [websocket.receive_json() for _ in range(6)]
    assert [event["type"] for event in events] == [
        "LISTENING_STOPPED",
        "PROCESSING_STARTED",
        "TRANSCRIPT_PARTIAL",
        "TRANSCRIPT_FINAL",
        "CAPABILITY_DEGRADED",
        "ERROR",
    ]


def test_disconnect_before_stop_session_does_not_raise(client):
    # Send START_SESSION and receive its one event, then let the with block
    # exit without sending STOP_SESSION. The server must handle the client
    # disconnect (via session.close()); no exception may escape this test.
    with client.websocket_connect(AUTHENTICATED_PATH) as websocket:
        websocket.send_json({"type": "START_SESSION", "session_type": "AI_CHAT"})
        websocket.receive_json()
