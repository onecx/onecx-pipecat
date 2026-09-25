# onecx-pipecat

This package implements the transport-neutral, fake-backed `AI_CHAT`
voice-session spine described in issue #597: an authenticated WebSocket entry
point that lets a client explicitly start and stop a voice session, exchanges a
defined set of commands and events, keeps session state and microphone state
independent, propagates tenant/principal context into every adapter call, and
can be fully exercised end-to-end with deterministic in-memory fake adapters —
no AWS, MCP, or other OneCX module dependency.

## Install

```
pip install -e ".[test]"
```

## Run

```
uvicorn onecx_pipecat.app:app --reload
```

## Test

```
pytest -v
```

## Documentation

See the transport-neutral voice-session protocol reference:
[docs/modules/ROOT/pages/voice-session-protocol.adoc](docs/modules/ROOT/pages/voice-session-protocol.adoc)
