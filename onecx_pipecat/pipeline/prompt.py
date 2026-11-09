"""System prompt composition, mirroring the OneCX Java runtime.

See ``ScaffoldPromptComposer`` and ``RuntimeSkillService`` in
onecx-ai-provider-runtime: skills without a name or instruction are dropped,
the rest are sorted by name (case-insensitive) and advertised after the
scaffold system prompt.
"""

from __future__ import annotations

from onecx_pipecat.models import AgentConfigSnapshot, SkillSnapshot


def usable_skills(snapshot: AgentConfigSnapshot) -> list[SkillSnapshot]:
    """Skills with a name and instruction, trimmed and sorted by name (case-insensitive)."""
    skills = [
        skill.model_copy(
            update={
                "name": skill.name.strip(),
                "description": skill.description.strip(),
                "instruction": skill.instruction.strip(),
            }
        )
        for skill in snapshot.skills
    ]
    return sorted(
        (s for s in skills if s.name and s.instruction),
        key=lambda s: s.name.lower(),
    )


def compose_system_prompt(
    snapshot: AgentConfigSnapshot, ai_context: list[str] | None = None
) -> str:
    """Scaffold prompt, skills block and page context, joined by a blank line.

    The page context block mirrors ``ScaffoldPromptComposer`` in the Java runtime.
    """
    blocks: list[str] = []
    if snapshot.system_prompt.strip():
        blocks.append(snapshot.system_prompt.strip())
    skills = usable_skills(snapshot)
    if skills:
        lines = [f"- {s.name}: {s.description or s.name}" for s in skills]
        blocks.append(
            "Available skills:\n"
            + "\n".join(lines)
            + "\nActivate a relevant skill before applying its instructions."
        )
    if ai_context:
        blocks.append("AI context:\n" + "\n".join(ai_context))
    return "\n\n".join(blocks)
