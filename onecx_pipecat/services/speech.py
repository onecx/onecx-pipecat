"""Pipecat STT/TTS services for the configured speech provider and pinned language."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onecx_pipecat.config import AwsSpeechProfile, Settings, SpeechSettings
from onecx_pipecat.services.inmemory import InMemorySTT, InMemoryTTS

if TYPE_CHECKING:
    from pipecat.audio.vad.vad_analyzer import VADAnalyzer
    from pipecat.processors.frame_processor import FrameProcessor

# AWS Transcribe drops a stream that receives no audio for a while; a voice
# session idles between utterances, so silence is sent to keep it open.
_STT_KEEPALIVE_SECS = 5.0


class UnsupportedLanguageError(Exception):
    """Raised when a pinned language has no AWS speech profile configured."""


def resolve_profile(settings: SpeechSettings, language_code: str) -> AwsSpeechProfile:
    """Map a pinned language to its AWS Transcribe/Polly profiles.

    An empty code falls back to the configured default language. A language
    that is not a key in ``aws_profiles`` is unsupported and raises
    ``UnsupportedLanguageError`` rather than guessing a profile.
    """
    code = language_code or settings.default_language
    profile = settings.aws_profiles.get(code)
    if profile is None:
        raise UnsupportedLanguageError(
            f"no AWS speech profile for language {code!r}; "
            f"configured: {sorted(settings.aws_profiles)}"
        )
    return profile


def build_stt_service(settings: Settings, language: str) -> FrameProcessor:
    speech = settings.speech
    if speech.stt_provider == "aws":
        from pipecat.services.aws.stt import AWSTranscribeSTTService

        profile = resolve_profile(speech, language)
        return AWSTranscribeSTTService(
            region=speech.region,
            sample_rate=speech.sample_rate_hz,
            keepalive_timeout=_STT_KEEPALIVE_SECS,
            settings=AWSTranscribeSTTService.Settings(language=profile.transcribe_language),
        )
    return InMemorySTT()


def build_tts_service(settings: Settings, language: str) -> FrameProcessor:
    speech = settings.speech
    if speech.tts_provider == "aws":
        from pipecat.services.aws.tts import AWSPollyTTSService

        profile = resolve_profile(speech, language)
        return AWSPollyTTSService(
            region=speech.region,
            sample_rate=speech.sample_rate_hz,
            settings=AWSPollyTTSService.Settings(
                voice=profile.polly_voice, language=profile.transcribe_language
            ),
        )
    return InMemoryTTS(sample_rate=speech.sample_rate_hz)


def build_vad_analyzer(settings: Settings) -> VADAnalyzer | None:
    """Silero VAD for server-side turn detection; None when the client delimits turns."""
    speech = settings.speech
    if speech.turn_detection != "vad":
        return None
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    return SileroVADAnalyzer(sample_rate=speech.sample_rate_hz)
