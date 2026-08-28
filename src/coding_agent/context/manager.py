from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from coding_agent.domain import (
    AssistantExchange,
    ConversationExchange,
    ToolContinuationExchange,
    UserExchange,
)

PressureLevel = Literal["safe", "soft", "hard"]
PersistCheckpoint = Callable[[str, object], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ContextStatus:
    used_tokens: int
    context_window: int
    max_output_tokens: int
    soft_limit: int
    hard_limit: int
    level: PressureLevel


@dataclass(frozen=True, slots=True)
class ContextBudget:
    context_window: int
    max_output_tokens: int
    auto_ratio: float = 0.8

    def __post_init__(self) -> None:
        if self.context_window <= 0 or not 0 < self.max_output_tokens < self.context_window:
            raise ValueError("context window and output reservation are invalid")
        if not 0 < self.auto_ratio < 1:
            raise ValueError("auto compaction ratio must be between zero and one")

    def status(self, used_tokens: int) -> ContextStatus:
        hard_limit = self.context_window - self.max_output_tokens
        soft_limit = math.floor(hard_limit * self.auto_ratio)
        if used_tokens >= hard_limit:
            level: PressureLevel = "hard"
        elif used_tokens >= soft_limit:
            level = "soft"
        else:
            level = "safe"
        return ContextStatus(
            used_tokens=max(used_tokens, 0),
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            level=level,
        )


class TokenEstimator:
    """Conservative local estimate; Provider usage remains the authoritative value."""

    def estimate(self, exchanges: Sequence[ConversationExchange]) -> int:
        characters = sum(len(_exchange_text(exchange)) for exchange in exchanges)
        return max(1, math.ceil(characters / 3)) if exchanges else 0


@dataclass(frozen=True, slots=True)
class CompactionCheckpoint:
    reason: str
    retained_from: int
    before_tokens: int
    after_tokens: int
    summary: str
    projected: tuple[ConversationExchange, ...]


class ContextManager:
    def __init__(
        self,
        budget: ContextBudget,
        estimator: TokenEstimator,
        *,
        retained_exchanges: int = 6,
    ) -> None:
        if retained_exchanges < 1:
            raise ValueError("retained_exchanges must be positive")
        self._budget = budget
        self._estimator = estimator
        self._retained_exchanges = retained_exchanges
        self._checkpoint: CompactionCheckpoint | None = None

    def status(self, history: Sequence[ConversationExchange]) -> ContextStatus:
        return self._budget.status(self._estimator.estimate(self.project(history)))

    def project(self, history: Sequence[ConversationExchange]) -> tuple[ConversationExchange, ...]:
        if self._checkpoint is None:
            return tuple(history)
        return (
            UserExchange(self._checkpoint.summary),
            *tuple(history[self._checkpoint.retained_from :]),
        )

    def prepare(
        self,
        history: Sequence[ConversationExchange],
        *,
        reason: str,
    ) -> CompactionCheckpoint | None:
        retained_from = max(0, len(history) - self._retained_exchanges)
        if retained_from == 0:
            return None
        compacted = tuple(history[:retained_from])
        summary = _summarize(compacted)
        projected: tuple[ConversationExchange, ...] = (
            UserExchange(summary),
            *tuple(history[retained_from:]),
        )
        return CompactionCheckpoint(
            reason=reason,
            retained_from=retained_from,
            before_tokens=self._estimator.estimate(history),
            after_tokens=self._estimator.estimate(projected),
            summary=summary,
            projected=projected,
        )

    async def compact(
        self,
        history: Sequence[ConversationExchange],
        *,
        reason: str,
        persist: PersistCheckpoint,
    ) -> CompactionCheckpoint | None:
        candidate = self.prepare(history, reason=reason)
        if candidate is None:
            return None
        await persist(
            "compaction_completed",
            {
                "reason": candidate.reason,
                "retained_from": candidate.retained_from,
                "before_tokens": candidate.before_tokens,
                "after_tokens": candidate.after_tokens,
                "summary": candidate.summary,
            },
        )
        self._checkpoint = candidate
        return candidate


def _summarize(exchanges: Sequence[ConversationExchange]) -> str:
    lines = ["Earlier conversation summary (deterministic local compaction):"]
    for exchange in exchanges:
        rendered = _exchange_text(exchange).strip().replace("\x00", "")
        if rendered:
            lines.append(f"- {rendered[:500]}")
    return "\n".join(lines)


def _exchange_text(exchange: ConversationExchange) -> str:
    if isinstance(exchange, UserExchange):
        return f"User: {exchange.content}"
    if isinstance(exchange, AssistantExchange):
        tools = ", ".join(call.name for call in exchange.tool_uses)
        return f"Assistant: {exchange.text}" + (f" [tools: {tools}]" if tools else "")
    if isinstance(exchange, ToolContinuationExchange):
        results = "; ".join(
            f"{result.tool_use_id}={'error' if result.is_error else 'ok'}: {result.content}"
            for result in exchange.results
        )
        return f"Assistant tools: {results}"
    raise TypeError(f"Unsupported exchange: {type(exchange).__name__}")
