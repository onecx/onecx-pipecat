"""FastAPI application entry point for the onecx-pipecat voice-session spine."""

from fastapi import FastAPI

from .transport.websocket import build_voice_ws_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="onecx-pipecat")
    app.include_router(build_voice_ws_router())
    return app


app = create_app()
