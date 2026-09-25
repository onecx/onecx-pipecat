"""Shared pytest fixtures and constants for onecx-pipecat tests."""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from onecx_pipecat.adapters import (
    DegradedFakeAssistantAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
)
from onecx_pipecat.app import create_app
from onecx_pipecat.transport.websocket import build_voice_ws_router


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def degraded_client() -> TestClient:
    app = FastAPI()
    app.include_router(
        build_voice_ws_router(
            adapter_factory=lambda: (
                FakeSpeechToTextAdapter(),
                FakeTextToSpeechAdapter(),
                DegradedFakeAssistantAdapter(),
            )
        )
    )
    return TestClient(app)


AUTHENTICATED_PATH = "/ws/voice-sessions/AI_CHAT" + "?tenant_id=tenant-a" + "&principal_id=user-1"
