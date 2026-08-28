from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from coding_agent.domain import (
    AssistantExchange,
    ConversationExchange,
    RedactedThinkingBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    UserExchange,
)

PressureLevel = Literal["safe", "soft", "hard"]
PersistCheckpoint = Callable[[str, object], Awaitable[None]]
SummarizeContext = Callable[[tuple[ConversationExchange, ...]], Awaitable[str]]

SUMMARY_FIELDS = (
    "task_goal",
    "user_constraints",
    "decisions",
    "files_read",
    "files_modified",
    "commands_and_results",
    "verification_status",
    "known_failures",
    "pending_work",
)


@dataclass(frozen=True, slots=True)
class ContextStatus:
    used_tokens: int
    context_window: int
    max_output_tokens: int
    soft_limit: int
    hard_limit: int
    level: PressureLevel
    usage_source: Literal["estimated"] = "estimated"
    last_provider_input_tokens: int | None = None


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

    def __init__(self) -> None:
        self._calibration = 1.0

    def estimate(self, exchanges: Sequence[ConversationExchange]) -> int:
        characters = sum(len(_exchange_text(exchange)) for exchange in exchanges)
        baseline = max(1, math.ceil(characters / 3)) if exchanges else 0
        return math.ceil(baseline * self._calibration)

    def calibrate(self, exchanges: Sequence[ConversationExchange], actual_tokens: int) -> None:
        if actual_tokens <= 0:
            return
        characters = sum(len(_exchange_text(exchange)) for exchange in exchanges)
        baseline = max(1, math.ceil(characters / 3))
        self._calibration = min(4.0, max(0.25, actual_tokens / baseline))


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
        excluded_thinking_indices: frozenset[int] = frozenset(),
    ) -> None:
        if retained_exchanges < 1:
            raise ValueError("retained_exchanges must be positive")
        self._budget = budget
        self._estimator = estimator
        self._retained_exchanges = retained_exchanges
        self._excluded_thinking_indices = excluded_thinking_indices
        self._checkpoint: CompactionCheckpoint | None = None
        self._last_provider_input_tokens: int | None = None

    def status(self, history: Sequence[ConversationExchange]) -> ContextStatus:
        status = self._budget.status(self._estimator.estimate(self.project(history)))
        return ContextStatus(
            used_tokens=status.used_tokens,
            context_window=status.context_window,
            max_output_tokens=status.max_output_tokens,
            soft_limit=status.soft_limit,
            hard_limit=status.hard_limit,
            level=status.level,
            last_provider_input_tokens=self._last_provider_input_tokens,
        )

    def record_provider_usage(
        self,
        request: Sequence[ConversationExchange],
        usage: Mapping[str, int],
    ) -> None:
        actual = usage.get("input_tokens")
        if actual is None:
            return
        self._last_provider_input_tokens = actual
        self._estimator.calibrate(request, actual)

    def project(self, history: Sequence[ConversationExchange]) -> tuple[ConversationExchange, ...]:
        if self._checkpoint is None:
            projected = tuple(enumerate(history))
        else:
            projected = (
                (None, UserExchange(self._checkpoint.summary)),
                *tuple(
                    enumerate(
                        history[self._checkpoint.retained_from :], self._checkpoint.retained_from
                    )
                ),
            )
        return tuple(
            filtered_exchange
            for index, exchange in projected
            if (
                filtered_exchange := (
                    _without_thinking(exchange)
                    if index in self._excluded_thinking_indices
                    else exchange
                )
            )
            is not None
        )

    def prepare(
        self,
        history: Sequence[ConversationExchange],
        *,
        reason: str,
        summary: str | None = None,
    ) -> CompactionCheckpoint | None:
        retained_from = max(0, len(history) - self._retained_exchanges)
        if retained_from == 0:
            return None
        compacted = tuple(history[:retained_from])
        summary = summary or _summarize(compacted)
        projected: tuple[ConversationExchange, ...] = (
            UserExchange(summary),
            *tuple(history[retained_from:]),
        )
        before_tokens = self._estimator.estimate(history)
        after_tokens = self._estimator.estimate(projected)
        if after_tokens >= before_tokens:
            return None
        return CompactionCheckpoint(
            reason=reason,
            retained_from=retained_from,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summary=summary,
            projected=projected,
        )

    async def compact(
        self,
        history: Sequence[ConversationExchange],
        *,
        reason: str,
        persist: PersistCheckpoint,
        summarize: SummarizeContext | None = None,
    ) -> CompactionCheckpoint | None:
        retained_from = max(0, len(history) - self._retained_exchanges)
        if retained_from == 0:
            return None
        await persist(
            "compaction_started",
            {
                "reason": reason,
                "before_tokens": self._estimator.estimate(history),
            },
        )
        candidate: CompactionCheckpoint | None = None
        strategy = "deterministic"
        if summarize is not None:
            try:
                summary_input = tuple(history[:retained_from])
                if self._excluded_thinking_indices:
                    summary_input = tuple(
                        filtered_exchange
                        for index, exchange in enumerate(summary_input)
                        if (
                            filtered_exchange := (
                                _without_thinking(exchange)
                                if index in self._excluded_thinking_indices
                                else exchange
                            )
                        )
                        is not None
                    )
                generated = await summarize(summary_input)
                if not _valid_structured_summary(generated):
                    raise ValueError("Provider returned an invalid context summary")
                candidate = self.prepare(history, reason=reason, summary=generated)
                if candidate is None:
                    raise ValueError("Provider summary does not reduce context usage")
                strategy = "provider"
            except Exception:
                await persist(
                    "compaction_strategy_failed",
                    {"reason": reason, "strategy": "provider"},
                )
        if candidate is None:
            candidate = self.prepare(history, reason=reason)
        if candidate is None:
            await persist("compaction_failed", {"reason": reason})
            return None
        await persist(
            "compaction_completed",
            {
                "reason": candidate.reason,
                "strategy": strategy,
                "retained_from": candidate.retained_from,
                "before_tokens": candidate.before_tokens,
                "after_tokens": candidate.after_tokens,
                "summary": candidate.summary,
            },
        )
        self._checkpoint = candidate
        return candidate

    def restore(
        self,
        history: Sequence[ConversationExchange],
        payload: Mapping[str, object],
    ) -> CompactionCheckpoint:
        reason = payload.get("reason")
        retained_from = payload.get("retained_from")
        before_tokens = payload.get("before_tokens")
        after_tokens = payload.get("after_tokens")
        summary = payload.get("summary")
        if (
            not isinstance(reason, str)
            or not isinstance(retained_from, int)
            or isinstance(retained_from, bool)
            or not 0 < retained_from <= len(history)
            or not isinstance(before_tokens, int)
            or not isinstance(after_tokens, int)
            or not isinstance(summary, str)
            or not summary.strip()
        ):
            raise ValueError("Invalid compaction checkpoint")
        projected: tuple[ConversationExchange, ...] = (
            UserExchange(summary),
            *tuple(history[retained_from:]),
        )
        checkpoint = CompactionCheckpoint(
            reason=reason,
            retained_from=retained_from,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summary=summary,
            projected=projected,
        )
        self._checkpoint = checkpoint
        return checkpoint


def _summarize(exchanges: Sequence[ConversationExchange]) -> str:
    excerpts: list[str] = []
    for exchange in exchanges:
        rendered = _exchange_text(exchange).strip().replace("\x00", "")
        if rendered:
            excerpts.append(rendered[:120])
    transcript = " | ".join(excerpts)[:140] or "unknown"
    values = {
        "task_goal": excerpts[0][:80] if excerpts else "unknown",
        "user_constraints": "unknown",
        "decisions": transcript,
        "files_read": "unknown",
        "files_modified": "unknown",
        "commands_and_results": "unknown",
        "verification_status": "unknown",
        "known_failures": "unknown",
        "pending_work": "continue from retained exchanges",
    }
    lines = ["context_summary_version: 1", "strategy: deterministic"]
    lines.extend(f"{field}: {values[field]}" for field in SUMMARY_FIELDS)
    return "\n".join(lines)[:1600]


def _valid_structured_summary(summary: str) -> bool:
    if not summary.strip():
        return False
    present = {line.partition(":")[0].strip() for line in summary.splitlines() if ":" in line}
    return all(field in present for field in SUMMARY_FIELDS)


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


def _without_thinking(exchange: ConversationExchange) -> ConversationExchange | None:
    if isinstance(exchange, UserExchange):
        return exchange
    if isinstance(exchange, ToolContinuationExchange):
        assistant = _assistant_without_thinking(exchange.assistant)
        if assistant is None:
            raise ValueError("Tool continuation lost its tool-use blocks during projection")
        return ToolContinuationExchange(assistant, exchange.results)
    return _assistant_without_thinking(exchange)


def _assistant_without_thinking(exchange: AssistantExchange) -> AssistantExchange | None:
    blocks = tuple(
        block
        for block in exchange.blocks
        if not isinstance(block, (ThinkingBlock, RedactedThinkingBlock))
    )
    if not blocks:
        return None
    return AssistantExchange(blocks, exchange.stop_reason, exchange.usage)
