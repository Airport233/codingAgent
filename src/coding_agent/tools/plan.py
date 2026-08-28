from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from coding_agent.plan import PlanState, PlanStatus, PlanStep
from coding_agent.tools.base import ToolOutput


class PlanStepInput(BaseModel):
    step: str = Field(min_length=1, max_length=200)
    status: PlanStatus

    @field_validator("step")
    @classmethod
    def normalize_step(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("step must not be blank")
        return normalized


class UpdatePlanInput(BaseModel):
    explanation: str | None = Field(default=None, max_length=500)
    plan: list[PlanStepInput] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_plan(self) -> UpdatePlanInput:
        steps = tuple(PlanStep(item.step, item.status) for item in self.plan)
        candidate = PlanState()
        candidate.update(steps)
        return self


class UpdatePlanTool:
    name = "update_plan"
    description = (
        "Create or replace a short task plan. Keep at most one step in_progress and update "
        "statuses as work advances."
    )
    input_model = UpdatePlanInput

    def __init__(self, state: PlanState) -> None:
        self._state = state

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = UpdatePlanInput.model_validate(arguments)
        steps = tuple(PlanStep(item.step, item.status) for item in parsed.plan)
        self._state.update(steps)
        serialized = [step.as_dict() for step in steps]
        active = sum(step.status == "in_progress" for step in steps)
        completed = sum(step.status == "completed" for step in steps)
        return ToolOutput(
            f"Plan updated: {active}/{len(steps)} active, {completed}/{len(steps)} completed",
            {"explanation": parsed.explanation or "", "plan": serialized},
        )
