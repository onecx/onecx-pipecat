"""Adapter interfaces and deterministic fake implementations."""

from .base import SpeechToTextAdapter, TextToSpeechAdapter, AssistantAdapter, TranscriptChunk
from .fakes import (
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
    FakeAssistantAdapter,
    DegradedFakeAssistantAdapter,
)

__all__ = [
    "SpeechToTextAdapter",
    "TextToSpeechAdapter",
    "AssistantAdapter",
    "TranscriptChunk",
    "FakeSpeechToTextAdapter",
    "FakeTextToSpeechAdapter",
    "FakeAssistantAdapter",
    "DegradedFakeAssistantAdapter",
]
