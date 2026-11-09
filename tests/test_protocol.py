from onecx_pipecat.protocol import (
    MAX_AI_CONTEXT_ENTRIES,
    MAX_AI_CONTEXT_ENTRY_CHARS,
    PROTOCOL_VERSION,
    ContextUpdate,
    StartSession,
    decode_command,
)

_START = {
    "type": "start_session",
    "agent_id": "agent",
    "chat_id": "chat-1",
    "apm_principal_token": "tok",
}


def test_protocol_version_is_three():
    assert PROTOCOL_VERSION == 3


def test_start_session_without_ai_context_defaults_to_empty():
    command = decode_command(_START)
    assert isinstance(command, StartSession)
    assert command.ai_context == []


def test_start_session_carries_ai_context():
    command = decode_command({**_START, "ai_context": ['{"app":"x"}']})
    assert isinstance(command, StartSession)
    assert command.ai_context == ['{"app":"x"}']


def test_context_update_decodes():
    command = decode_command({"type": "context_update", "ai_context": ["a", "b"]})
    assert command == ContextUpdate(ai_context=["a", "b"])


def test_context_update_without_entries_clears_context():
    assert decode_command({"type": "context_update"}) == ContextUpdate(ai_context=[])


def test_malformed_ai_context_is_rejected():
    assert decode_command({**_START, "ai_context": "not-a-list"}) is None
    assert decode_command({**_START, "ai_context": [1]}) is None
    assert decode_command({"type": "context_update", "ai_context": None}) is None


def test_too_many_ai_context_entries_are_rejected():
    entries = ["x"] * (MAX_AI_CONTEXT_ENTRIES + 1)
    assert decode_command({"type": "context_update", "ai_context": entries}) is None


def test_oversized_ai_context_entry_is_rejected():
    entry = "x" * (MAX_AI_CONTEXT_ENTRY_CHARS + 1)
    assert decode_command({**_START, "ai_context": [entry]}) is None
