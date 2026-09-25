"""FastAPI router exposing the authenticated ``/ws/voice-sessions/{session_type}`` endpoint.

The endpoint authenticates before accepting, rejects unsupported session
types, constructs a ``VoiceSession`` from an injectable adapter factory,
relays commands and events, and guarantees cleanup on disconnect.

Inbound frames are read by a background task into a queue, so a command
such as ``INTERRUPT`` sent by the client while a turn is still streaming is
captured immediately. The relay drives each turn's event stream one event at
a time and checks the queue for a pending ``INTERRUPT`` at every event
boundary; when one is found it is dispatched inline and the turn stops, so
the interruption lands before the turn's remaining ``ASSISTANT_*`` events.
"""

import asyncio
from typing import Callable
from uuid import uuid4

from fastapi import APIRouter
from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..adapters import (
    AssistantAdapter,
    FakeAssistantAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
    SpeechToTextAdapter,
    TextToSpeechAdapter,
)
from ..auth import AuthenticationError, authenticate
from ..context import SessionContext
from ..protocol import (
    CommandType,
    ErrorEvent,
    InboundCommand,
    InterruptCommand,
    SessionStoppedEvent,
    parse_inbound,
    serialize_outbound,
)
from ..session import VoiceSession

AdapterFactory = Callable[[], tuple[SpeechToTextAdapter, TextToSpeechAdapter, AssistantAdapter]]

# Queue sentinel pushed by the receive task when the client disconnects.
_DISCONNECTED = object()


def default_adapter_factory() -> tuple[SpeechToTextAdapter, TextToSpeechAdapter, AssistantAdapter]:
    return (
        FakeSpeechToTextAdapter(),
        FakeTextToSpeechAdapter(),
        FakeAssistantAdapter(),
    )


def build_voice_ws_router(adapter_factory: AdapterFactory = default_adapter_factory) -> APIRouter:
    """Build the voice-session WebSocket router with an injectable adapter factory."""
    router = APIRouter()

    @router.websocket("/ws/voice-sessions/{session_type}")
    async def voice_session_endpoint(websocket: WebSocket, session_type: str) -> None:
        try:
            tenant_id, principal_id = authenticate(websocket)
        except AuthenticationError:
            await websocket.close(code=4401)
            return

        if session_type != "AI_CHAT":
            await websocket.close(code=4400)
            return

        await websocket.accept()
        context = SessionContext(
            session_id=str(uuid4()),
            tenant_id=tenant_id,
            principal_id=principal_id,
            session_type="AI_CHAT",
        )
        stt, tts, assistant = adapter_factory()
        session = VoiceSession(context, stt, tts, assistant)

        command_queue: asyncio.Queue = asyncio.Queue()

        async def receive_commands() -> None:
            # Read inbound frames concurrently with turn processing so a
            # command (e.g. INTERRUPT) sent mid-turn is captured and queued
            # immediately rather than waiting for the turn to finish.
            while True:
                try:
                    raw = await websocket.receive_json()
                except WebSocketDisconnect:
                    command_queue.put_nowait(_DISCONNECTED)
                    return
                try:
                    command = parse_inbound(raw)
                except ValidationError as exc:
                    # Malformed frames yield a typed error event; queue it so
                    # its ERROR event is emitted in place of the bad frame.
                    command_queue.put_nowait(
                        ErrorEvent(code="INVALID_COMMAND", message=str(exc), fatal=False)
                    )
                    continue
                command_queue.put_nowait(command)

        receive_task = asyncio.create_task(receive_commands())

        async with session:
            try:
                while True:
                    item = await command_queue.get()
                    if item is _DISCONNECTED:
                        break
                    if isinstance(item, ErrorEvent):
                        await websocket.send_json(serialize_outbound(item))
                        continue
                    if _is_turn_command(item):
                        stopped = await _run_turn_with_interrupt(
                            websocket, session, item, command_queue
                        )
                    else:
                        stopped = await _send_command_events(websocket, session, item)
                    if stopped:
                        break
                # The only way to exit the loop normally is the session
                # stopping (or the client disconnecting); end the connection.
                await websocket.close()
            except WebSocketDisconnect:
                pass
            finally:
                # Cancel the receive task and await it so the loop does not
                # keep a task pending ("Task was destroyed but it is pending").
                # This is best-effort cleanup; any exception here (cancellation
                # or a mid-read disconnect) is not fatal.
                receive_task.cancel()
                try:
                    await receive_task
                except Exception:
                    pass

    return router


def _is_turn_command(command: InboundCommand) -> bool:
    """A turn command is one whose handling streams a full assistant turn.

    ``STOP_LISTENING`` synchronously runs the turn (transcript, assistant,
    TTS); it is the only command that needs an interrupt check mid-stream.
    """
    return command.type == CommandType.STOP_LISTENING


def _try_take_interrupt(command_queue: asyncio.Queue) -> InboundCommand | None:
    """Non-blockingly look for an ``INTERRUPT`` at the front of the queue.

    Returns the interrupt, or ``None`` (leaving the queue unchanged) when the
    front item is not an interrupt.
    """
    try:
        item = command_queue.get_nowait()
    except asyncio.QueueEmpty:
        return None
    if isinstance(item, InterruptCommand):
        return item
    # Not an interrupt: put it back so it is processed in order after the
    # turn completes.
    command_queue.put_nowait(item)
    return None


async def _send_command_events(
    websocket: WebSocket,
    session: VoiceSession,
    command: InboundCommand,
) -> bool:
    """Send every event produced by one command.

    Returns ``True`` when the session stopped (so the caller can end the
    connection). Expected session errors are already converted into events by
    ``handle_command``; anything else (e.g. the client went away mid-stream)
    simply ends this command.
    """
    stopped = False
    try:
        async for event in session.handle_command(command):
            await websocket.send_json(serialize_outbound(event))
            if isinstance(event, SessionStoppedEvent):
                stopped = True
    except Exception:
        pass
    return stopped


async def _run_turn_with_interrupt(
    websocket: WebSocket,
    session: VoiceSession,
    turn_command: InboundCommand,
    command_queue: asyncio.Queue,
) -> bool:
    """Stream a turn's events one at a time, honoring a queued ``INTERRUPT``.

    Each event boundary gives the queue a chance to deliver an ``INTERRUPT``.
    When one is found it is dispatched inline (producing ``TURN_INTERRUPTED``
    and completing the turn in the session state machine) and the turn's
    remaining events are abandoned. Returns ``True`` when the session stopped.
    """
    stopped = False
    agen = session.handle_command(turn_command)
    try:
        while True:
            try:
                event = await agen.__anext__()
            except StopAsyncIteration:
                break
            await websocket.send_json(serialize_outbound(event))
            if isinstance(event, SessionStoppedEvent):
                stopped = True
                break
            interrupt = _try_take_interrupt(command_queue)
            if interrupt is not None:
                # Interrupt while the turn is still streaming: dispatch it now.
                stopped = await _send_command_events(websocket, session, interrupt)
                break
    except Exception:
        pass
    finally:
        # Abandon the paused turn generator cleanly if an interrupt cut it
        # short; a fully-drained generator is already exhausted.
        try:
            await agen.aclose()
        except Exception:
            pass
    return stopped
