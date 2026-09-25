"""Unit tests for the session and microphone state machines."""

import pytest

from onecx_pipecat.errors import (
    IllegalMicrophoneTransitionError,
    IllegalSessionTransitionError,
)
from onecx_pipecat.session_state import (
    MicrophoneState,
    MicrophoneStateMachine,
    SessionState,
    SessionStateMachine,
)


def test_session_state_machine_full_happy_path():
    fsm = SessionStateMachine()
    assert fsm.apply("start") == SessionState.READY
    assert fsm.apply("begin_listening") == SessionState.LISTENING
    assert fsm.apply("begin_processing") == SessionState.PROCESSING
    assert fsm.apply("begin_responding") == SessionState.RESPONDING
    assert fsm.apply("finish_turn") == SessionState.READY
    assert fsm.apply("stop") == SessionState.STOPPED


def test_session_state_machine_rejects_out_of_order_trigger():
    fsm = SessionStateMachine()
    with pytest.raises(IllegalSessionTransitionError):
        fsm.apply("begin_processing")
    assert fsm.state == SessionState.CREATED


def test_microphone_state_machine_happy_path():
    mic = MicrophoneStateMachine()
    assert mic.apply("enable") == MicrophoneState.LISTENING
    assert mic.apply("suspend") == MicrophoneState.SUSPENDED
    assert mic.apply("resume") == MicrophoneState.LISTENING
    assert mic.apply("disable") == MicrophoneState.MUTED


def test_microphone_state_machine_rejects_illegal_resume():
    mic = MicrophoneStateMachine()
    with pytest.raises(IllegalMicrophoneTransitionError):
        mic.apply("resume")
    assert mic.state == MicrophoneState.MUTED


def test_session_and_microphone_states_are_independent():
    fsm = SessionStateMachine()
    mic = MicrophoneStateMachine()
    fsm.apply("start")
    fsm.apply("begin_listening")
    assert mic.state == MicrophoneState.MUTED
    mic.apply("enable")
    assert fsm.state == SessionState.LISTENING
