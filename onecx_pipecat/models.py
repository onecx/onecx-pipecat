"""Typed domain value models for the Pipecat foundation.

State names mirror the vocabulary in
``prototypes/voice-turn-sequence-prototype.html`` so the eventual session
logic can use the same strings without translation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Principal(BaseModel):
    """An authenticated actor bound to a voice session."""

    subject: str
    tenant_id: str
    roles: list[str] = Field(default_factory=list)


class PermissionAction(StrEnum):
    """Canonical OneCX permission vocabulary.

    ``ALWAYS_ALLOW`` is the only action a voice session may execute a tool
    with; ``ALWAYS_ASK`` and ``DENY`` tools are not advertised or invokable.
    """

    DENY = "DENY"
    ALWAYS_ASK = "ALWAYS_ASK"
    ALWAYS_ALLOW = "ALWAYS_ALLOW"


class ToolRule(BaseModel):
    """A resolved per-tool permission (ai-provider ``ToolRuleConfiguration``)."""

    tool_name: str
    permission: PermissionAction
    description: str = ""


class McpServerSnapshot(BaseModel):
    """One MCP server of the pinned agent configuration (ai-provider ``McpServerConfiguration``)."""

    id: str = ""
    name: str
    url: str
    credential_ref: str = ""
    unlisted_tool_permission: PermissionAction = PermissionAction.DENY
    tool_rules: list[ToolRule] = Field(default_factory=list)


class ProviderSnapshot(BaseModel):
    """The LLM provider of the pinned agent (ai-provider ``ProviderConfiguration``)."""

    id: str = ""
    name: str = ""
    type: str
    llm_url: str = ""
    auth_mode: str = ""
    credential_ref: str = ""


class ModelSnapshot(BaseModel):
    """The LLM model of the pinned agent (ai-provider ``ModelConfiguration``)."""

    id: str = ""
    name: str = ""
    model_identifier: str
    # Opaque JSON string of provider-specific model parameters (ai-provider
    # ``modelConfig``; not named ``model_config``, which Pydantic reserves).
    model_params: str = ""
    communication_mode: str = ""


class SkillSnapshot(BaseModel):
    """A scaffold skill of the pinned agent (ai-provider ``SkillConfiguration``)."""

    id: str = ""
    name: str = ""
    description: str = ""
    instruction: str = ""
    position: int = 0
    source: str = ""


class AgentConfigSnapshot(BaseModel):
    """The agent configuration pinned for one voice session.

    Fetched once at session start (ai-provider ``AgentConfiguration``) and
    pinned for the whole session; configuration changes apply to the next
    session. Tool policy is already resolved by ai-provider per MCP server.
    """

    agent_id: str
    schema_version: str
    agent_config_version: str = ""
    system_prompt: str = ""
    additional_prompt: str = ""
    mcp_servers: list[McpServerSnapshot] = Field(default_factory=list)
    # Pinned voice language (``voice.languageCode``); "" when the configuration
    # does not specify one, so speech falls back to the configured default.
    language_code: str = ""
    agent_name: str = ""
    provider: ProviderSnapshot | None = None
    model: ModelSnapshot | None = None
    skills: list[SkillSnapshot] = Field(default_factory=list)


class AgentToolSnapshot(BaseModel):
    """A tool discovered on one of the snapshot's MCP servers."""

    tool_name: str
    server_name: str
    server_url: str = ""
    tool_description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class VoiceEntryKind(StrEnum):
    HUMAN = "HUMAN"
    ASSISTANT = "ASSISTANT"
    ASSISTANT_CHECKPOINT = "ASSISTANT_CHECKPOINT"
    TOOL_PROPOSAL = "TOOL_PROPOSAL"
    TOOL_DECISION = "TOOL_DECISION"
    TOOL_RESULT = "TOOL_RESULT"


class VoiceEntryStatus(StrEnum):
    """Terminal lifecycle state of a durable voice entry.

    A voice entry is written exactly once, already in a terminal state
    (``COMPLETED`` when the turn ran to its natural end, ``INTERRUPTED`` when
    the user barged in and cut the assistant output short). Interim streaming
    states are carried by the wire protocol, never by a persisted entry.
    """

    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"


class VoiceEntry(BaseModel):
    """A durable conversation entry (visible or hidden checkpoint)."""

    kind: VoiceEntryKind
    text: str
    hidden: bool = False
    idempotency_key: str | None = None
    # Terminal state the entry was finalized in (see ``VoiceEntryStatus``).
    status: VoiceEntryStatus = VoiceEntryStatus.COMPLETED
    # Stamped by the session from its pinned snapshot before persistence.
    agent_config_version: str = ""


class PermissionDecision(BaseModel):
    """The result of a permission evaluation for a tool call."""

    action: PermissionAction
    reason: str = ""


class PermissionDecisionStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    CONSUMED = "CONSUMED"


class ToolProposal(BaseModel):
    """A protected tool call awaiting a permission decision."""

    proposal_id: str
    tool_name: str
    server_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: PermissionDecisionStatus = PermissionDecisionStatus.PENDING
    reason: str = ""


class TranscriptState(StrEnum):
    NONE = "NONE"
    PARTIAL = "PARTIAL"
    FINAL = "FINAL"
    FAILED = "FAILED"


class AudioState(StrEnum):
    NONE = "NONE"
    SYNTHESIZING = "SYNTHESIZING"
    PLAYING = "PLAYING"
    STOPPED = "STOPPED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
