from __future__ import annotations

from pydantic import BaseModel, Field

from coding_agent.skills import SkillSnapshot
from coding_agent.tools.base import RecoverableToolError, Tool, ToolOutput


class ActivateSkillInput(BaseModel):
    name: str = Field(min_length=1)


class ActivateSkillTool:
    name = "activate_skill"
    description = (
        "Activate one available Agent Skill for the current task. Use the skill metadata "
        "in system instructions to choose a name."
    )
    input_model = ActivateSkillInput

    def __init__(self, skills: SkillSnapshot) -> None:
        self._skills = skills

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = ActivateSkillInput.model_validate(arguments)
        skill = self._skills.get(parsed.name)
        if skill is None:
            raise RecoverableToolError(f"Unknown skill: {parsed.name}")
        return ToolOutput(
            f"Activated {skill.name}; its SKILL.md instructions will be loaded next.",
            {"skill_name": skill.name, "source": skill.source},
        )


class ReadSkillResourceInput(BaseModel):
    skill_name: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ReadSkillResourceTool:
    name = "read_skill_resource"
    description = (
        "Read one UTF-8 scripts/, references/, or assets/ file from an installed Agent Skill, "
        "using a path relative to that skill's directory."
    )
    input_model = ReadSkillResourceInput

    def __init__(self, skills: SkillSnapshot) -> None:
        self._skills = skills

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = ReadSkillResourceInput.model_validate(arguments)
        skill = self._skills.get(parsed.skill_name)
        if skill is None:
            raise RecoverableToolError(f"Unknown skill: {parsed.skill_name}")
        normalized = parsed.path.replace("\\", "/")
        if not normalized.startswith(("references/", "scripts/", "assets/")):
            raise RecoverableToolError(
                "skill resources must be inside references/, scripts/, or assets/"
            )
        try:
            content = skill.read_resource(parsed.path)
        except (OSError, UnicodeError, ValueError) as error:
            raise RecoverableToolError(str(error)) from error
        return ToolOutput(content)


class SkillToolSource:
    source_id = "agent-skills"

    def __init__(self, skills: SkillSnapshot) -> None:
        self._skills = skills

    async def list_tools(self) -> tuple[Tool, ...]:
        return (ActivateSkillTool(self._skills), ReadSkillResourceTool(self._skills))
