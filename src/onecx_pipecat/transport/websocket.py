"""FastAPI router exposing the authenticated ``/ws/voice-sessions/{session_type}`` endpoint.

The endpoint authenticates before accepting, rejects unsupported session
types, constructs a ``VoiceSession`` from an injectable adapter factory,
relays commands and events, and guarantees cleanup on disconnect.
"""

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
    ErrorEvent,
    SessionStoppedEvent,
    parse_inbound,
    serialize_outbound,
)
from ..session import VoiceSession

AdapterFactory = Callable[[], tuple[SpeechToTextAdapter, TextToSpeechAdapter, AssistantAdapter]]


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

        async with session:
            try:
                while True:
                    raw = await websocket.receive_json()
                    try:
                        command = parse_inbound(raw)
                    except ValidationError as exc:
                        await websocket.send_json(
                            serialize_outbound(
                                ErrorEvent(
                                    code="INVALID_COMMAND", message=str(exc), fatal=False
                                )
                            )
                        )
                        continue
                    stopped = False
                    async for event in session.handle_command(command):
                        await websocket.send_json(serialize_outbound(event))
                        if isinstance(event, SessionStoppedEvent):
                            stopped = True
                    if stopped:
                        break
                # The only way to exit the loop normally is the session stopping;
                # end the connection now that the session is over.
                await websocket.close()
            except WebSocketDisconnect:
                pass

    return router
