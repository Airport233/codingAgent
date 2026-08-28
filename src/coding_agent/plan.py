from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

type PlanStatus = Literal["pending", "in_progress", "completed"]
_PLAN_STATUSES = frozenset({"pending", "in_progress", "completed"})


@dataclass(frozen=True, slots=True)
class PlanStep:
    step: str
    status: PlanStatus

    def as_dict(self) -> dict[str, str]:
        return {"step": self.step, "status": self.status}


class PlanState:
    def __init__(self, steps: tuple[PlanStep, ...] = ()) -> None:
        self._steps: tuple[PlanStep, ...] = ()
        if steps:
            self.update(steps)

    def snapshot(self) -> tuple[PlanStep, ...]:
        return self._steps

    def update(self, steps: tuple[PlanStep, ...]) -> None:
        normalized = tuple(PlanStep(item.step.strip(), item.status) for item in steps)
        _validate_steps(normalized)
        self._steps = normalized

    def restore(self, payload: object) -> None:
        if not isinstance(payload, list):
            raise ValueError("stored plan must be a list")
        restored: list[PlanStep] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("stored plan step must be an object")
            step = item.get("step")
            status = item.get("status")
            if not isinstance(step, str) or not isinstance(status, str):
                raise ValueError("stored plan fields must be strings")
            if status not in _PLAN_STATUSES:
                raise ValueError("stored plan status is invalid")
            restored.append(PlanStep(step, cast(PlanStatus, status)))
        if restored:
            self.update(tuple(restored))
        else:
            self._steps = ()

    def render_context(self) -> str:
        if not self._steps:
            return ""
        lines = ["<current_plan>"]
        lines.extend(f"- [{step.status}] {step.step}" for step in self._steps)
        lines.append("</current_plan>")
        return "\n".join(lines)


def _validate_steps(steps: tuple[PlanStep, ...]) -> None:
    if len(steps) > 8:
        raise ValueError("plan may contain at most 8 steps")
    normalized: set[str] = set()
    active = 0
    for item in steps:
        if item.status not in _PLAN_STATUSES:
            raise ValueError("plan status is invalid")
        text = item.step.strip()
        if not text or len(text) > 200:
            raise ValueError("plan step must contain 1 to 200 characters")
        folded = text.casefold()
        if folded in normalized:
            raise ValueError("plan steps must be unique")
        normalized.add(folded)
        active += item.status == "in_progress"
    if active > 1:
        raise ValueError("plan may contain at most one in_progress step")
