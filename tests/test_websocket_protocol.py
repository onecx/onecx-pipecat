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


def test_interrupt_during_live_turn_over_websocket(client):
    # The client must be able to interrupt a turn while it is actually
    # running over the wire: send STOP_LISTENING (which starts the turn)
    # immediately followed by INTERRUPT, without waiting for the turn's
    # events. The TURN_INTERRUPTED event must be produced before the turn's
    # ASSISTANT_* events complete.
    with client.websocket_connect(AUTHENTICATED_PATH) as websocket:
        websocket.send_json({"type": "START_SESSION", "session_type": "AI_CHAT"})
        first = websocket.receive_json()
        assert first["type"] == "SESSION_READY"
        websocket.send_json({"type": "START_LISTENING"})
        second = websocket.receive_json()
        assert second["type"] == "LISTENING_STARTED"
        websocket.send_json({"type": "AUDIO_CHUNK", "audio_base64": "AAA=", "sequence": 1})
        websocket.send_json({"type": "STOP_LISTENING"})
        # Immediately send the interrupt, before any turn event is consumed.
        websocket.send_json({"type": "INTERRUPT"})

        # Collect the turn's events up to and including the interruption.
        turn_types = []
        while True:
            event = websocket.receive_json()
            turn_types.append(event["type"])
            if event["type"] == "TURN_INTERRUPTED":
                break

        # TURN_INTERRUPTED is only produced while the session is PROCESSING
        # or RESPONDING; after the turn finished it would instead be an
        # ILLEGAL_SESSION_TRANSITION ERROR. Receiving it proves the interrupt
        # was read and dispatched while the turn was still running.
        assert "TURN_INTERRUPTED" in turn_types
        # The turn's ASSISTANT_* events must not have completed: a full turn
        # emits 3 ASSISTANT_TEXT + 2 ASSISTANT_AUDIO_CHUNK = 5 assistant
        # events, but the interrupt stops the turn mid-stream, so strictly
        # fewer than 5 assistant events were produced before it.
        assistant_events = sum(
            1 for t in turn_types if t in ("ASSISTANT_TEXT", "ASSISTANT_AUDIO_CHUNK")
        )
        assert assistant_events < 5

        # The connection remains operational after the interruption.
        websocket.send_json({"type": "STOP_SESSION"})
        final = websocket.receive_json()
        assert final["type"] == "SESSION_STOPPED"


def test_interrupted_turn_ends_connection_operatively(client):
    # After an INTERRUPTed turn the session is back in READY; the connection
    # must remain operational and still honor STOP_SESSION.
    with client.websocket_connect(AUTHENTICATED_PATH) as websocket:
        websocket.send_json({"type": "START_SESSION", "session_type": "AI_CHAT"})
        websocket.receive_json()  # SESSION_READY
        websocket.send_json({"type": "START_LISTENING"})
        websocket.receive_json()  # LISTENING_STARTED
        websocket.send_json({"type": "AUDIO_CHUNK", "audio_base64": "AAA=", "sequence": 1})
        websocket.send_json({"type": "STOP_LISTENING"})
        websocket.send_json({"type": "INTERRUPT"})
        # Drain the (interrupted) turn up to and including TURN_INTERRUPTED.
        while True:
            event = websocket.receive_json()
            if event["type"] == "TURN_INTERRUPTED":
                break
        websocket.send_json({"type": "STOP_SESSION"})
        assert websocket.receive_json()["type"] == "SESSION_STOPPED"


def test_interrupt_without_active_turn_yields_typed_error_over_websocket(client):
    # INTERRUPT while no turn is processing/responding is an illegal
    # session transition and must surface a typed ERROR event over the wire.
    with client.websocket_connect(AUTHENTICATED_PATH) as websocket:
        websocket.send_json({"type": "START_SESSION", "session_type": "AI_CHAT"})
        assert websocket.receive_json()["type"] == "SESSION_READY"
        websocket.send_json({"type": "INTERRUPT"})
        event = websocket.receive_json()
    assert event["type"] == "ERROR"
    assert event["code"] == "ILLEGAL_SESSION_TRANSITION"
