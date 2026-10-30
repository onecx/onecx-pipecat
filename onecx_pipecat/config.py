"""Typed application configuration.

Loaded from the environment with a stable prefix so deployments can override
any field without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    # The Keycloak variables below are the unprefixed names the platform
    # provisions; the ``ONECX_AUTH__`` form stays accepted for each of them.
    model_config = SettingsConfigDict(env_prefix="ONECX_AUTH__", populate_by_name=True)

    # Keycloak base URL and realm (``KEYCLOAK_SERVER_URL``/``KEYCLOAK_REALM``).
    # When both are set and ``issuer`` is not, the issuer is
    # ``{server_url}/realms/{realm}``.
    server_url: str = Field(
        default="",
        validation_alias=AliasChoices("KEYCLOAK_SERVER_URL", "ONECX_AUTH__SERVER_URL"),
    )
    realm: str = Field(
        default="",
        validation_alias=AliasChoices("KEYCLOAK_REALM", "ONECX_AUTH__REALM"),
    )
    issuer: str = "https://auth.example"
    token_claim_subject: str = "sub"
    # Keycloak JWKS used to verify ``apm-principal-token``: an embedded JSON
    # document or a URL; empty fetches ``{issuer}/protocol/openid-connect/certs``.
    jwks: str = ""
    # Pipecat's own OIDC client credentials, used to authenticate outbound
    # calls to OneCX services the way a BFF does (apm-principal-token is also
    # propagated to act as the user).
    client_id: str = Field(
        default="onecx-pipecat",
        validation_alias=AliasChoices("KEYCLOAK_CLIENT_ID", "ONECX_AUTH__CLIENT_ID"),
    )
    client_secret: str = Field(
        default="",
        validation_alias=AliasChoices("KEYCLOAK_CLIENT_SECRET", "ONECX_AUTH__CLIENT_SECRET"),
    )
    # Empty uses ``{issuer}/protocol/openid-connect/token``.
    token_endpoint: str = ""

    @model_validator(mode="after")
    def _derive_issuer(self) -> AuthSettings:
        if "issuer" not in self.model_fields_set and self.server_url and self.realm:
            self.issuer = f"{self.server_url.rstrip('/')}/realms/{self.realm}"
        return self


class AgentConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONECX_AGENT_CONFIG__")

    default_agent_id: str = "onecx-agent"
    default_snapshot_version: int = 1
    # Base URL of onecx-ai-provider-svc (``GET /v1/agents/{id}/configuration``).
    base_url: str = "http://onecx-ai-provider-svc:8080"


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONECX_CHAT__")

    # Base URL of onecx-chat-svc, used to load a chat by id as the user.
    base_url: str = "http://onecx-chat-svc:8080"
    # Newest chat messages loaded into the LLM context at session start.
    history_max_messages: int = 50


class AwsSpeechProfile(BaseModel):
    """The AWS speech profiles for one language.

    A profile pairs the AWS Transcribe language code with the Amazon Polly
    voice the assistant speaks with, so a single pinned language maps to both
    STT and TTS deployments.
    """

    transcribe_language: str
    polly_voice: str = "Joanna"


class SpeechSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONECX_SPEECH__")

    stt_provider: str = "inmemory"
    tts_provider: str = "inmemory"
    sample_rate_hz: int = 16000
    # "vad": Pipecat detects speech (Silero VAD) and ends the user turn after a
    # pause, so the client only streams audio. "client": the client delimits
    # turns with user_started_speaking / user_stopped_speaking (push-to-talk).
    turn_detection: Literal["vad", "client"] = "vad"
    # AWS region the Transcribe and Polly profiles live in (aws provider only).
    region: str = "us-east-1"
    # Fallback when a pinned agent configuration does not specify a language.
    default_language: str = "en"
    # Language code -> {transcribe_language, polly_voice}. A pinned language
    # that is not a key here is unsupported and produces a typed error.
    aws_profiles: dict[str, AwsSpeechProfile] = Field(
        default_factory=lambda: {
            "en": AwsSpeechProfile(transcribe_language="en-US", polly_voice="Joanna"),
        }
    )


class PersistenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONECX_PERSISTENCE__")

    # "chat" saves to onecx-chat-svc; "inmemory" is for tests and local runs.
    backend: Literal["chat", "inmemory"] = "chat"
    max_entries_per_session: int = 512


class PermissionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONECX_PERMISSION__")

    # Base URL of onecx-permission-svc (``POST /v1/permissions/user/{productName}/{appId}``).
    base_url: str = "http://onecx-permission-svc:8080"
    # The OneCX permission that grants voice: ``voice_action`` on ``voice_resource``
    # of application ``voice_app_id`` in product ``voice_product_name``. The
    # onecx-chat-ui Helm chart declares it.
    voice_product_name: str = "onecx-chat"
    voice_app_id: str = "onecx-chat-ui"
    voice_resource: str = "VOICE"
    voice_action: str = "USE"


class McpSettings(BaseSettings):
    """Operator-level MCP knobs; server URLs come from the agent configuration."""

    model_config = SettingsConfigDict(env_prefix="ONECX_MCP__")

    timeout_seconds: float = 30.0


class CredentialsSettings(BaseSettings):
    """Operator-provisioned secrets, keyed by the ``credentialRef`` of the agent configuration."""

    model_config = SettingsConfigDict(env_prefix="ONECX_CREDENTIALS__")

    # credentialRef (e.g. ``credential://provider/{id}/api-key``) -> secret.
    refs: dict[str, SecretStr] = Field(default_factory=dict)


class LlmSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONECX_LLM__")

    # Test/local-dev only: ``inmemory`` replaces the agent's provider with a
    # deterministic fake. Empty (the default) keeps production fail-closed.
    provider_override: str = ""


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONECX_PIPELINE__")

    # Seconds without bot/user speech activity before the pipeline ends; None disables.
    idle_timeout_secs: float | None = None


class Settings(BaseSettings):
    """Root configuration for the Pipecat application.

    Every adapter's knobs live on a nested model so configuration stays
    typed and the environment mapping stays explicit.
    """

    model_config = SettingsConfigDict(
        env_prefix="ONECX_", env_nested_delimiter="__", populate_by_name=True
    )

    app_name: str = "onecx-pipecat"
    host: str = "127.0.0.1"
    # ``SERVER_PORT`` is the name the platform provisions; ``ONECX_PORT`` stays accepted.
    port: int = Field(default=8080, validation_alias=AliasChoices("SERVER_PORT", "ONECX_PORT"))
    log_level: str = "INFO"
    default_tenant_id: str = "local"

    auth: AuthSettings = Field(default_factory=AuthSettings)
    agent_config: AgentConfigSettings = Field(default_factory=AgentConfigSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    speech: SpeechSettings = Field(default_factory=SpeechSettings)
    persistence: PersistenceSettings = Field(default_factory=PersistenceSettings)
    permission: PermissionSettings = Field(default_factory=PermissionSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    credentials: CredentialsSettings = Field(default_factory=CredentialsSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide Settings instance."""
    return Settings()
