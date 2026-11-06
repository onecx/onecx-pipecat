"""FastAPI application for the Pipecat foundation."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import asdict
from importlib import metadata
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from onecx_pipecat.container import AppContainer
from onecx_pipecat.protocol import ErrorEvent, Event, decode_command
from onecx_pipecat.session import SessionState, VoiceSession

# WebSocket close code for an unauthenticated or otherwise rejected connection.
_WS_POLICY_VIOLATION = 1008
# Close reason sent when the browser origin is not allowed to open a session.
_WS_ORIGIN_NOT_ALLOWED = "origin_not_allowed"

logger = logging.getLogger(__name__)


def _pipecat_version() -> str:
    try:
        return metadata.version("pipecat-ai")
    except metadata.PackageNotFoundError:
        return "unknown"


def _origin_allowed(websocket: WebSocket, policy: str) -> bool:
    """Return True when the browser origin may open a voice session.

    ``off`` skips the check. ``deny`` rejects every connection (fail-closed).
    ``same-host`` compares the ``Origin`` hostname to the host the WebSocket
    request was addressed to (the ``Host`` header), ignoring scheme and port;
    a missing, ``null`` or unparseable origin is rejected.
    """
    if policy == "off":
        return True
    if policy == "deny":
        return False
    origin = websocket.headers.get("origin")
    if not origin or origin == "null":
        return False
    try:
        origin_host = urlparse(origin).hostname
    except ValueError:
        return False
    if not origin_host:
        return False
    request_host = websocket.headers.get("host", "").split(":", 1)[0]
    if not request_host:
        request_host = websocket.url.hostname or ""
    return origin_host.lower() == request_host.lower()


def create_app(container: AppContainer, expiry_check_seconds: float = 5.0) -> FastAPI:
    app = FastAPI(title=container.settings.app_name)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "OK",
            "app": container.settings.app_name,
            "pipecat_version": _pipecat_version(),
            "adapters": asdict(container.adapters()),
        }

    @app.websocket("/ws/voice")
    async def voice(websocket: WebSocket) -> None:
        # Tokens belong in the first message, not the URL, and the endpoint takes
        # no query parameters at all: any of them is rejected.
        if websocket.query_params:
            await websocket.close(code=_WS_POLICY_VIOLATION)
            return

        await websocket.accept()
        # Reject disallowed browser origins before any session state is created
        # so the client receives a clear close reason instead of a silent drop.
        if not _origin_allowed(websocket, container.settings.voice.origin_policy):
            logger.info(
                "Rejected voice WebSocket: origin %r not allowed by policy %r",
                websocket.headers.get("origin"),
                container.settings.voice.origin_policy,
            )
            await websocket.close(code=_WS_POLICY_VIOLATION, reason=_WS_ORIGIN_NOT_ALLOWED)
            return
        session = VoiceSession(uuid4().hex, container)

        async def send(events: list[Event]) -> None:
            for event in events:
                await websocket.send_json(event.model_dump())

        async def watch_expiry() -> None:
            # An idle session must end at token expiry, not only when audio arrives.
            while True:
                await asyncio.sleep(expiry_check_seconds)
                if events := session.check_expiry():
                    await session.end_pipeline(events, _WS_POLICY_VIOLATION)
                    return

        watcher: asyncio.Task | None = None
        try:
            # Handshake: only a valid ``start_session`` moves on to the pipeline.
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = None
                command = decode_command(payload) if isinstance(payload, dict) else None
                if command is None:
                    await send([ErrorEvent(code="bad_command", message="malformed command")])
                    continue
                await send(await session.dispatch(command))
                if session.state is SessionState.READY:
                    break
                if session.state is SessionState.CLOSED:
                    await websocket.close(
                        code=_WS_POLICY_VIOLATION if session.start_failed else 1000
                    )
                    return
            watcher = asyncio.create_task(watch_expiry())
            await session.run_pipeline(websocket)
        except WebSocketDisconnect:
            return
        finally:
            await session.release()
            if watcher is not None:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher

    return app
