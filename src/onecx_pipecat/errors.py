"""Typed exception hierarchy shared by the state machines, adapters, and
orchestrator.

Each exception carries a stable ``error_code`` that is written into the
``code`` field of the wire-level ``ERROR`` event.
"""


class VoiceSessionError(Exception):
    """Base class for all voice-session errors."""

    error_code: str = "SESSION_ERROR"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class IllegalSessionTransitionError(VoiceSessionError):
    """Raised when a trigger is applied while the session is in a state that
    does not define that trigger."""

    error_code = "ILLEGAL_SESSION_TRANSITION"

    def __init__(self, current_state: str, attempted_trigger: str) -> None:
        self.current_state = current_state
        self.attempted_trigger = attempted_trigger
        super().__init__(
            f"cannot apply '{attempted_trigger}' while in state '{current_state}'"
        )


class IllegalMicrophoneTransitionError(VoiceSessionError):
    """Raised when a trigger is applied while the microphone is in a state
    that does not define that trigger."""

    error_code = "ILLEGAL_MICROPHONE_TRANSITION"

    def __init__(self, current_state: str, attempted_trigger: str) -> None:
        self.current_state = current_state
        self.attempted_trigger = attempted_trigger
        super().__init__(
            f"cannot apply '{attempted_trigger}' while in state '{current_state}'"
        )


class AdapterFailureError(VoiceSessionError):
    """Raised when an adapter reports a capability failure."""

    error_code = "ADAPTER_FAILURE"

    def __init__(self, adapter_name: str, message: str) -> None:
        self.adapter_name = adapter_name
        super().__init__(f"{adapter_name} failed: {message}")
