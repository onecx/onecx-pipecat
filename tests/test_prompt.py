from onecx_pipecat.models import AgentConfigSnapshot, SkillSnapshot
from onecx_pipecat.pipeline.prompt import compose_system_prompt, usable_skills


def _snap(prompt: str = "be brief", skills: list[SkillSnapshot] | None = None):
    return AgentConfigSnapshot(
        agent_id="a", schema_version="1", system_prompt=prompt, skills=skills or []
    )


def test_prompt_without_skills_is_the_scaffold_prompt():
    assert compose_system_prompt(_snap("  be brief \n")) == "be brief"


def test_ai_context_block_follows_the_prompt_like_the_java_runtime():
    prompt = compose_system_prompt(_snap("be brief"), ['{"a":1}', '{"b":2}'])
    assert prompt == 'be brief\n\nAI context:\n{"a":1}\n{"b":2}'


def test_empty_ai_context_adds_no_block():
    assert compose_system_prompt(_snap("be brief"), []) == "be brief"


def test_empty_everything_is_empty():
    assert compose_system_prompt(_snap("")) == ""


def test_skills_block_is_sorted_and_appended():
    skills = [
        SkillSnapshot(name="beta", description="Second", instruction="do b"),
        SkillSnapshot(name="Alpha", description="", instruction="do a"),
    ]
    assert compose_system_prompt(_snap(skills=skills)) == (
        "be brief\n\n"
        "Available skills:\n- Alpha: Alpha\n- beta: Second\n"
        "Activate a relevant skill before applying its instructions."
    )


def test_skills_block_without_scaffold_prompt():
    skills = [SkillSnapshot(name="a", instruction="x", description="d")]
    assert compose_system_prompt(_snap("", skills)).startswith("Available skills:\n- a: d")


def test_blank_name_or_instruction_is_dropped_and_values_trimmed():
    skills = [
        SkillSnapshot(name=" ", instruction="x"),
        SkillSnapshot(name="a", instruction="  "),
        SkillSnapshot(name=" b ", description=" d ", instruction=" run "),
    ]
    usable = usable_skills(_snap(skills=skills))
    assert [(s.name, s.description, s.instruction) for s in usable] == [("b", "d", "run")]


def test_no_usable_skills_means_no_block():
    skills = [SkillSnapshot(name="a", instruction="")]
    assert compose_system_prompt(_snap(skills=skills)) == "be brief"
