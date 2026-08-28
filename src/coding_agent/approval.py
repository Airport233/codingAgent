from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from coding_agent.domain import ToolUseBlock

ApprovalMode = Literal["auto", "ask", "deny"]
ApprovalAction = Literal["allow", "ask", "deny"]
ApprovalDecision = Literal["allow_once", "allow_session", "deny"]


class ApprovalPolicy(Protocol):
    def evaluate(self, call: ToolUseBlock) -> ApprovalAction: ...

    def remember(self, call: ToolUseBlock, decision: ApprovalDecision) -> None: ...


@dataclass(slots=True)
class ConfigurableApprovalPolicy:
    mode: ApprovalMode = "auto"
    guarded_tools: frozenset[str] = frozenset()
    _session_allowed: set[str] = field(default_factory=set, init=False, repr=False)

    def evaluate(self, call: ToolUseBlock) -> ApprovalAction:
        if call.name not in self.guarded_tools or call.name in self._session_allowed:
            return "allow"
        return "allow" if self.mode == "auto" else self.mode

    def remember(self, call: ToolUseBlock, decision: ApprovalDecision) -> None:
        if decision == "allow_session":
            self._session_allowed.add(call.name)
