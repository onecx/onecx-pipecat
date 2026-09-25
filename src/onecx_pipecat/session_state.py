"""Independent finite-state machines for the voice session and microphone.

``SessionStateMachine`` and ``MicrophoneStateMachine`` share no state and
never reference each other; ``VoiceSession`` (``onecx_pipecat.session``)
holds exactly one instance of each.
"""

from enum import StrEnum

from .errors import IllegalMicrophoneTransitionError, IllegalSessionTransitionError


class SessionState(StrEnum):
    CREATED = "CREATED"
    READY = "READY"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    RESPONDING = "RESPONDING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class MicrophoneState(StrEnum):
    MUTED = "MUTED"
    LISTENING = "LISTENING"
    SUSPENDED = "SUSPENDED"


SESSION_TRANSITIONS: dict[SessionState, dict[str, SessionState]] = {
    SessionState.CREATED: {"start": SessionState.READY, "fault": SessionState.ERROR},
    SessionState.READY: {
        "begin_listening": SessionState.LISTENING,
        "stop": SessionState.STOPPED,
        "fault": SessionState.ERROR,
    },
    SessionState.LISTENING: {
        "begin_processing": SessionState.PROCESSING,
        "stop": SessionState.STOPPED,
        "fault": SessionState.ERROR,
    },
    SessionState.PROCESSING: {
        "begin_responding": SessionState.RESPONDING,
        "finish_turn": SessionState.READY,
        "stop": SessionState.STOPPED,
        "fault": SessionState.ERROR,
    },
    SessionState.RESPONDING: {
        "finish_turn": SessionState.READY,
        "stop": SessionState.STOPPED,
        "fault": SessionState.ERROR,
    },
    SessionState.STOPPED: {},
    SessionState.ERROR: {},
}


MIC_TRANSITIONS: dict[MicrophoneState, dict[str, MicrophoneState]] = {
    MicrophoneState.MUTED: {"enable": MicrophoneState.LISTENING},
    MicrophoneState.LISTENING: {
        "disable": MicrophoneState.MUTED,
        "suspend": MicrophoneState.SUSPENDED,
    },
    MicrophoneState.SUSPENDED: {
        "resume": MicrophoneState.LISTENING,
        "disable": MicrophoneState.MUTED,
    },
}


class SessionStateMachine:
    """State machine for the session lifecycle, driven by explicit triggers."""

    def __init__(self) -> None:
        self.state = SessionState.CREATED

    def apply(self, trigger: str) -> SessionState:
        next_state = SESSION_TRANSITIONS[self.state].get(trigger)
        if next_state is None:
            raise IllegalSessionTransitionError(self.state.value, trigger)
        self.state = next_state
        return self.state


class MicrophoneStateMachine:
    """State machine for the microphone, driven by explicit triggers."""

    def __init__(self) -> None:
        self.state = MicrophoneState.MUTED

    def apply(self, trigger: str) -> MicrophoneState:
        next_state = MIC_TRANSITIONS[self.state].get(trigger)
        if next_state is None:
            raise IllegalMicrophoneTransitionError(self.state.value, trigger)
        self.state = next_state
        return self.state
