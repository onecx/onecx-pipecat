"""Deterministic in-memory fake adapters.

These are the only adapter implementations used in this slice; no AWS SDK,
MCP client, or other OneCX module import appears in this module.
"""

import asyncio
from typing import AsyncIterable, AsyncIterator

from ..context import SessionContext
from ..errors import AdapterFailureError
from .base import AssistantAdapter, SpeechToTextAdapter, TextToSpeechAdapter, TranscriptChunk


class FakeSpeechToTextAdapter(SpeechToTextAdapter):
    """Emits a fixed two-chunk transcript (partial then final)."""

    def __init__(self) -> None:
        self.closed = False
        self.received_contexts: list[SessionContext] = []

    async def stream_transcripts(
        self, context: SessionContext, audio_chunks: AsyncIterable[bytes]
    ) -> AsyncIterator[TranscriptChunk]:
        self.received_contexts.append(context)
        async for _ in audio_chunks:
            pass
        await asyncio.sleep(0)
        yield TranscriptChunk("turn on the", False, 1)
        await asyncio.sleep(0)
        yield TranscriptChunk("turn on the lights", True, 2)

    async def aclose(self) -> None:
        self.closed = True


class FakeAssistantAdapter(AssistantAdapter):
    """Emits a fixed three-chunk assistant response."""

    def __init__(self) -> None:
        self.closed = False
        self.received_contexts: list[SessionContext] = []

    async def respond(
        self, context: SessionContext, transcript: str
    ) -> AsyncIterator[str]:
        self.received_contexts.append(context)
        yield "Sure,"
        await asyncio.sleep(0)
        yield " turning"
        await asyncio.sleep(0)
        yield " on the lights."

    async def aclose(self) -> None:
        self.closed = True


class FakeTextToSpeechAdapter(TextToSpeechAdapter):
    """Emits a fixed two-chunk synthesized audio payload."""

    def __init__(self) -> None:
        self.closed = False
        self.received_contexts: list[SessionContext] = []

    async def synthesize(
        self, context: SessionContext, text: str
    ) -> AsyncIterator[bytes]:
        self.received_contexts.append(context)
        yield b"\x00\x01"
        await asyncio.sleep(0)
        yield b"\x02\x03"

    async def aclose(self) -> None:
        self.closed = True


class DegradedFakeAssistantAdapter(AssistantAdapter):
    """Always fails, to exercise the degraded-capability path."""

    def __init__(self) -> None:
        self.closed = False

    async def respond(
        self, context: SessionContext, transcript: str
    ) -> AsyncIterator[str]:
        raise AdapterFailureError("assistant", "capability unavailable")
        yield ""

    async def aclose(self) -> None:
        self.closed = True
