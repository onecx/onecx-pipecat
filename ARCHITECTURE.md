# OneCX Pipecat: Architecture Overview

## Purpose

OneCX Pipecat is the foundation for **authenticated voice sessions** in OneCX.
It is a Python FastAPI service that exposes a single WebSocket voice endpoint.
Every external dependency (authentication, chat, agent configuration,
permissions, MCP tools, speech, persistence) sits behind a replaceable adapter,
so the service can run, be tested and be linted without cloud credentials.

> **Status:** each voice session runs one long-lived Pipecat pipeline
> (`FastAPIWebsocketTransport` → STT → LLM → TTS) driven by the agent's pinned
> configuration. Speech defaults to deterministic in-memory services until the
> AWS provider is selected; the LLM comes from the agent's provider.

## How Pipecat works (background)

[Pipecat](https://pipecat.ai) is an open-source Python framework for real-time
voice and multimodal agents.

| Concept | Description |
|---|---|
| **Frames** | Typed units of data flowing through the system: audio, transcriptions, LLM text, synthesized audio, and control signals (start, end, interrupt). |
| **Frame processors** | Single-purpose stages that consume frames and emit frames. |
| **Pipeline** | An ordered chain of processors, typically `transport in → STT → context aggregation → LLM → TTS → transport out`. |
| **Transport** | The real-time carrier: WebRTC, WebSocket or telephony. |
| **Services** | Pluggable STT, LLM and TTS vendors. The LLM can call tools, including MCP tools. |
| **Worker and runner** | `PipelineWorker` runs the pipeline; `WorkerRunner` manages its lifecycle. Interruptions (barge-in) are handled by the pipeline. |

OneCX Pipecat wraps this model with OneCX authentication, permissions, agent
configuration and tool policy.

## Component overview

Diagram: [component-overview.drawio](docs/diagrams/component-overview.drawio)

| Component | Responsibility |
|---|---|
| **Entrypoint** | Builds the container and the application, and starts uvicorn. |
| **HTTP / WebSocket app** | A health endpoint (status, Pipecat version, adapter and speech provider names) and the voice WebSocket. It rejects query parameters, runs the JSON handshake until a valid `start_session`, runs the expiry watcher, then hands the socket to the session's pipeline. |
| **Composition root (container)** | Owns one instance of each adapter and the credential resolver for the application's lifetime. |
| **Configuration** | Typed settings loaded from environment variables (prefix `ONECX_`, nested delimiter `__`), one group per adapter. |
| **Protocol** | The wire vocabulary: handshake and control commands, and typed events. |
| **Voice session** | Authenticates, pins the agent, builds the LLM, STT and TTS and the tool schema, enforces the MCP tool policy, then runs the pipeline. It contains no per-turn logic. |
| **Pipeline package** | `builder` (assembly), `serializer` (wire ⇄ frames), `processors` (session control, wire events), `persistence` (aggregator hooks), `tools` (MCP and skill functions), `prompt`, `runner`. |
| **Services package** | `llm` (OpenAI, Ollama, Anthropic from the pinned provider/model), `speech` (AWS Transcribe/Polly or in-memory), `credentials` (`credentialRef` → secret), `inmemory`. |
| **Domain models** | Principal, pinned agent configuration (provider, model, skills, MCP servers and tool rules), permission actions, conversation entries, tool proposals. |
| **Adapters** | Interfaces with implementations for each OneCX boundary (see below). |

### Pipeline

```
transport.input → session control → STT → transcript events → user aggregator
  → LLM → TTS → wire events → transport.output → assistant aggregator
```

Turns are detected on the server by default (`speech.turn_detection=vad`): a
Silero VAD in the user aggregator starts the user turn on speech, which
interrupts the bot, and ends it after a pause once a transcript arrived. The
client then only streams audio. With `turn_detection=client` the client
delimits turns (`user_started_speaking` / `user_stopped_speaking`, push-to-talk)
through Pipecat's external user turn strategies. Either way the server reports
`user_started_speaking` / `user_stopped_speaking`. Persistence hangs on the
aggregators' turn events and `turn_complete` is
sent only after the assistant entry was stored. Service `ErrorFrame`s become
non-fatal `stt_failed` / `llm_failed` / `tts_failed` events. See the voice
session documentation for the full protocol.

### Adapters

| Adapter | Implementation | Backing service and behaviour |
|---|---|---|
| **Authentication** | JWT verification | Verifies the user's `apm-principal-token` (signature, issuer, expiry) against the Keycloak JWKS and yields a principal. Can read expiry without enforcing it. |
| **Service authentication** | OIDC client credentials | Pipecat's own token for outbound OneCX calls, in the way a BFF does it. |
| **Agent configuration** | HTTP | Fetches the agent configuration from the AI provider service. Missing, invalid, unreachable or voice-disabled agents fail the session start. |
| **Chat** | HTTP | Loads a chat by id as the user and confirms the user is a member. |
| **Permission** | HTTP | Checks the user's existing OneCX voice permission. Any error counts as "no permission". |
| **MCP** | Streamable-HTTP MCP client | Discovers tools per server and invokes tools. Redirects are never followed, so user tokens cannot leak to another host. |
| **Tool policy** | Pure functions | Resolves each tool to a permission from its server's tool rules, else the server's default for unlisted tools. |
| **Persistence** | `HttpChatPersistence` (default), `InMemoryPersistence` | Keyed by chat id: saves HUMAN/ASSISTANT turns to onecx-chat-svc (`POST /internal/chats/{id}/messages`) and loads the chat history at start (`GET`). `ONECX_PERSISTENCE__BACKEND=inmemory` for tests and local runs. |

Speech and the LLM are Pipecat services built per session, not adapters.

The OneCX boundaries (authentication, agent configuration, chat, permission,
MCP) always use their real implementation. There is no in-memory fallback that
a deployment could silently run on. Only speech (by default), the opt-in
`inmemory` persistence backend and the test-only LLM override are in-memory.

## Session protocol

Handshake: `start_session` → `session_ready` (+ `degraded_capability`). After
that the pipeline's serializer carries:

| Client → server | Server → client |
|---|---|
| `audio` (base64 PCM16); with client turn detection also `user_started_speaking`, `user_stopped_speaking` | `transcript`, `user_started_speaking`, `user_stopped_speaking`, `assistant_text`, `assistant_audio`, `bot_started_speaking`, `bot_stopped_speaking`, `interrupted`, `turn_complete` |
| `refresh_token`, `stop` | `error` (typed code), `session_closed` (reason) |

Error codes: `authentication_failed`, `chat_not_accessible`,
`voice_not_permitted`, `config_unavailable`, `unsupported_language`,
`token_expired`, `illegal_state`, `principal_mismatch`, `bad_command`,
`stt_failed`, `llm_failed`, `tts_failed`, `persistence_failed`.

## Session lifecycle

### States

```
CREATED ──start ok──▶ READY ──stop / token expired──▶ CLOSED
   └────start fails (fail closed)─────────────────────▶ CLOSED
```

### Connection handling

1. The connection is rejected if it carries any query parameter. Tokens must
   never be in the URL.
2. The socket is accepted unauthenticated and a new session is created.
   Authentication is established by the first message.
3. A background watcher checks token expiry periodically, so an idle session
   still ends when its token expires: it sends `token_expired` and
   `session_closed`, stops the pipeline and closes with 1008.
4. Until a valid `start_session` arrives, a malformed or unknown message
   produces a `bad_command` error. Afterwards the pipeline owns the socket.
5. The session ends on `stop`, on token expiry, on a failed start, or on
   client disconnect. The pipeline worker and the watcher are always cancelled.

### Starting a session

Every step fails closed with a typed error, followed by `session_closed`.

1. **Authenticate** the token carried in the message.
2. **Load the chat** as the user and require that the user belongs to it.
3. **Check the voice permission.**
4. **Pin the agent configuration** for the whole session (provider, model,
   system prompt, skills, MCP servers). Configuration changes apply to the next
   session.
5. **Build the LLM and speech services.** Unusable provider, model or
   credential fails with `config_unavailable`; an unsupported language with
   `unsupported_language`.
6. **Discover MCP tools once** on the agent's own servers, forwarding the
   user's tokens.
7. **Filter the tools**: only tools resolved to `ALWAYS_ALLOW` are advertised
   (as Pipecat functions, plus `activate_skill` when skills exist).
   `ALWAYS_ASK` and `DENY` tools are withheld.
8. **Become ready**: emit `session_ready`, plus one `degraded_capability` per
   MCP server that failed. The session continues without that server's tools.

### While the session is ready

- **Expiry guard:** the watcher ends sessions with expired tokens.
- **Voice turn:** handled entirely by the pipeline (see above). Empty or
  noise-only input stays ephemeral. A turn persists one human and one
  assistant entry (`COMPLETED` or `INTERRUPTED`) and then emits `turn_complete`.
- **Token refresh:** the new token is verified and must belong to the same
  subject and tenant. A refresh can extend a session but never switch its user,
  because chat access and voice permission are only checked at start.
- **Tool invocation:** LLM function calls reach `VoiceSession.invoke_tool`; only
  tools in the session's allowed list can run, on servers from the pinned
  configuration, with the user's tokens forwarded.
- **Stop:** ends the session and returns the user to typed chat.

### Sequence

Diagram: [session-sequence.drawio](docs/diagrams/session-sequence.drawio)

## Security properties

- Tokens travel only in the first message, never in the URL.
- Every start step fails closed with a typed error.
- The agent configuration and the tool allow-list are fixed at session start.
- Pipecat's own service token is never sent to per-agent MCP servers. Only the
  user's tokens are.
- MCP calls never follow redirects.
- Token refresh cannot change the session's user.
- Tokens expire mid-session even when the client is idle.

## Quality and delivery

- **Tests:** async-first `pytest` suite covering the models, configuration,
  adapters, container, factory, session, application and WebSocket behaviour,
  using shared fakes for the external services.
- **Linting:** `ruff`, enforced by a pre-commit hook and CI.
- **CI:** tests, lint and documentation build on Python 3.11 and 3.12, without
  credentials.
- **Documentation:** an Antora module covering local setup, the configuration
  reference, adapter interfaces and the voice session.

## Known gaps and next steps

- Chat history carries only user/assistant text: an INTERRUPTED turn is stored
  as the spoken text without a marker (chat-svc `Message` has no status), and
  tool entries are not persisted. The history is loaded once at start, so
  messages typed into the chat during a voice session are not seen by it.
- Server VAD relies on the browser's echo cancellation; without it the bot's
  own voice can open a user turn and interrupt itself. A minimum-words start
  strategy would harden this.
- Support `ALWAYS_ASK` tool proposals and decisions. The models exist, but the
  flow does not.
- OAuth provider authentication for the LLM (only API-key credentials work).
- Add a WebRTC carrier.
