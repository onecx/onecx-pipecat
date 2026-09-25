"""Abstract adapter interfaces for the voice-session spine.

Every concrete adapter must implement these interfaces. Each method receives
the immutable ``SessionContext`` explicitly so tenant and principal
propagation is structural.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterable, AsyncIterator

from ..context import SessionContext


@dataclass(frozen=True)
class TranscriptChunk:
    """A single transcript fragment produced by a speech-to-text adapter."""

    text: str
    is_final: bool
    sequence: int


class SpeechToTextAdapter(ABC):
    """Converts streamed audio chunks into transcript chunks."""

    @abstractmethod
    async def stream_transcripts(
        self, context: SessionContext, audio_chunks: AsyncIterable[bytes]
    ) -> AsyncIterator[TranscriptChunk]:
        ...

    async def aclose(self) -> None:
        return None


class TextToSpeechAdapter(ABC):
    """Synthesizes assistant text into audio chunks."""

    @abstractmethod
    async def synthesize(
        self, context: SessionContext, text: str
    ) -> AsyncIterator[bytes]:
        ...

    async def aclose(self) -> None:
        return None


class AssistantAdapter(ABC):
    """Produces the assistant's text response for a finalized transcript."""

    @abstractmethod
    async def respond(
        self, context: SessionContext, transcript: str
    ) -> AsyncIterator[str]:
        ...

    async def aclose(self) -> None:
        return None
