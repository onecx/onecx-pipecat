from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import CancelWorkerFrame, InputTransportMessageFrame
from pipecat.processors.frame_processor import FrameDirection

from onecx_pipecat.app import create_app
from onecx_pipecat.pipeline.processors import SessionControl
from onecx_pipecat.protocol import ContextUpdate, ErrorEvent
from tests.fakes import fake_container
from tests.voice_helpers import _start, _until_bounded, wait_for_close


def _control(dispatch=None):
    emit = AsyncMock()
    dispatch = dispatch or AsyncMock(return_value=[])
    return SessionControl(dispatch=dispatch, emit=emit), dispatch, emit


async def _send(control, message):
    await control.process_frame(
        InputTransportMessageFrame(message=message), FrameDirection.DOWNSTREAM
    )


@pytest.mark.asyncio
async def test_context_update_is_dispatched_to_the_session():
    control, dispatch, emit = _control()

    await _send(control, {"type": "context_update", "ai_context": ["page: event 42"]})

    dispatch.assert_awaited_once_with(ContextUpdate(ai_context=["page: event 42"]))
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_events_from_a_context_update_are_emitted_to_the_client():
    rejected = ErrorEvent(code="illegal_state", message="session is not active")
    control, _, emit = _control(dispatch=AsyncMock(return_value=[rejected]))

    await _send(control, {"type": "context_update", "ai_context": []})

    emit.assert_awaited_once_with(rejected)


@pytest.mark.asyncio
async def test_malformed_context_update_is_rejected_as_bad_command():
    control, dispatch, emit = _control()

    await _send(control, {"type": "context_update", "ai_context": None})

    dispatch.assert_not_awaited()
    emit.assert_awaited_once_with(ErrorEvent(code="bad_command", message="malformed command"))


@pytest.mark.asyncio
async def test_start_session_is_not_accepted_on_the_control_path():
    control, dispatch, emit = _control()

    await _send(control, {"type": "start_session", "agent_id": "a", "chat_id": "c"})

    dispatch.assert_not_awaited()
    emit.assert_awaited_once_with(ErrorEvent(code="bad_command", message="malformed command"))


@pytest.mark.asyncio
async def test_stop_session_still_cancels_the_worker():
    control, dispatch, _ = _control()
    control.push_frame = AsyncMock()

    await _send(control, {"type": "stop"})

    dispatch.assert_awaited_once()
    control.push_frame.assert_awaited_once()
    frame, direction = control.push_frame.await_args.args
    assert isinstance(frame, CancelWorkerFrame)
    assert direction is FrameDirection.UPSTREAM


def test_context_update_on_a_live_session_produces_no_error():
    client = TestClient(create_app(fake_container(token="t-1")))
    with client.websocket_connect("/ws/voice") as ws:
        _start(ws)
        ws.send_json({"type": "context_update", "ai_context": ['{"event": "E2"}']})
        ws.send_json({"type": "stop"})
        events = _until_bounded(ws, "session_closed")
        wait_for_close(ws)

    assert not [event for event in events if event["kind"] == "error"]
