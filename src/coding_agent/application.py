from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence

from coding_agent.context import CompactionCheckpoint, ContextManager, ContextStatus
from coding_agent.domain import (
    AssistantExchange,
    Conversation,
    ConversationExchange,
    RedactedThinkingBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    UserExchange,
)
from coding_agent.events import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    AgentStarted,
    ContextUsageChanged,
    CoreEvent,
    TextDelta,
    ThinkingDelta,
    ThinkingFinished,
    ThinkingStarted,
    ToolFinished,
    ToolStarted,
    WarningRaised,
)
from coding_agent.memory.loader import ProjectMemoryLoader
from coding_agent.providers.base import (
    Provider,
    ProviderResponseFinished,
    ProviderTextDelta,
    ProviderThinkingDelta,
)
from coding_agent.sessions.base import SessionStore
from coding_agent.tools.dispatcher import ToolDispatcher


class AgentApplication:
    def __init__(
        self,
        provider: Provider,
        dispatcher: ToolDispatcher,
        sessions: SessionStore,
        max_steps: int = 20,
        memory_loader: ProjectMemoryLoader | None = None,
        initial_exchanges: Sequence[ConversationExchange] = (),
        context_manager: ContextManager | None = None,
        context_reprojected: bool = False,
        display_redactor: Callable[[object], object] | None = None,
    ) -> None:
        self._provider = provider
        self._dispatcher = dispatcher
        self._sessions = sessions
        self._max_steps = max_steps
        self._memory_loader = memory_loader
        self._last_memory_digest: str | None = None
        self._conversation = Conversation(list(initial_exchanges))
        self._context_manager = context_manager
        self._context_reprojected = context_reprojected
        self._display_redactor = display_redactor or (lambda value: value)
        self._closed = False

    async def close_session(self) -> None:
        if self._closed:
            return
        await self._sessions.append("session_closed", {})
        self._closed = True

    def context_status(self) -> ContextStatus | None:
        if self._context_manager is None:
            return None
        system_instructions = self._load_system_instructions()
        return self._context_manager.status(
            self._conversation.snapshot(),
            self._supplemental_characters(system_instructions),
        )

    async def compact_context(self, reason: str = "manual") -> CompactionCheckpoint | None:
        if self._context_manager is None:
            return None
        history = self._conversation.snapshot()
        supplemental_characters = self._supplemental_characters(self._load_system_instructions())
        summarize = (
            None
            if self._context_manager.status(history, supplemental_characters).level == "hard"
            else self._request_context_summary
        )
        return await self._context_manager.compact(
            history,
            reason=reason,
            persist=self._sessions.append,
            summarize=summarize,
            supplemental_characters=supplemental_characters,
        )

    def _load_system_instructions(self) -> str | None:
        if self._memory_loader is None:
            return None
        return self._memory_loader.load().rendered or None

    def _supplemental_characters(self, system_instructions: str | None) -> int:
        tools = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in self._dispatcher.catalog.specs
        ]
        serialized_tools = json.dumps(tools, ensure_ascii=False, sort_keys=True)
        return len(system_instructions or "") + len(serialized_tools)

    async def _request_context_summary(self, exchanges: tuple[ConversationExchange, ...]) -> str:
        instruction = UserExchange(
            "Summarize the preceding conversation as plain text with exactly these fields, "
            "one per line: task_goal, user_constraints, decisions, files_read, "
            "files_modified, commands_and_results, verification_status, known_failures, "
            "pending_work. Preserve concrete paths, commands, results, constraints, and "
            "unfinished work. Do not call tools."
        )
        response: AssistantExchange | None = None
        async for event in self._provider.stream((*exchanges, instruction), (), None):
            if isinstance(event, ProviderResponseFinished):
                response = event.exchange
        if response is None or response.stop_reason != "end_turn":
            raise RuntimeError("Provider did not complete the context summary")
        return response.text

    async def run(self, prompt: str) -> AsyncIterator[CoreEvent]:
        user_exchange = UserExchange(content=prompt)
        self._conversation.exchanges.append(user_exchange)
        await self._sessions.append("user_exchange", user_exchange)
        yield AgentStarted(prompt=prompt)
        if self._context_reprojected:
            yield WarningRaised(
                "Context was reprojected for the selected model; prior-model thinking was omitted"
            )
            self._context_reprojected = False

        for _step in range(self._max_steps):
            response: AssistantExchange | None = None
            system_instructions = self._load_system_instructions()
            if self._memory_loader is not None:
                memory = self._memory_loader.load()
                if memory.digest != self._last_memory_digest:
                    await self._sessions.append(
                        "memory_snapshot_changed",
                        {
                            "digest": memory.digest,
                            "entries": [
                                {
                                    "source": entry.source,
                                    "priority": entry.priority,
                                    "content": entry.content,
                                }
                                for entry in memory.entries
                            ],
                        },
                    )
                    self._last_memory_digest = memory.digest
            if self._context_manager is not None:
                supplemental_characters = self._supplemental_characters(system_instructions)
                status = self._context_manager.status(
                    self._conversation.snapshot(), supplemental_characters
                )
                if status.level in {"soft", "hard"}:
                    try:
                        await self.compact_context(reason="auto")
                    except Exception:
                        yield WarningRaised(
                            "Context compaction failed; continuing with the original context"
                        )
                request_history = self._context_manager.project(self._conversation.snapshot())
                status = self._context_manager.status(
                    self._conversation.snapshot(), supplemental_characters
                )
                yield ContextUsageChanged(
                    used_tokens=status.used_tokens,
                    context_window=status.context_window,
                    level=status.level,
                )
                if status.level == "hard":
                    message = "Context remains above the safe request limit"
                    await self._sessions.append("turn_failed", {"message": message})
                    yield AgentFailed(message=message)
                    return
            else:
                request_history = self._conversation.snapshot()
            thinking_active = False
            try:
                async for event in self._provider.stream(
                    request_history,
                    self._dispatcher.catalog.specs,
                    system_instructions,
                ):
                    if isinstance(event, ProviderThinkingDelta):
                        if not thinking_active:
                            thinking_active = True
                            yield ThinkingStarted()
                        yield ThinkingDelta(text=event.thinking)
                    elif isinstance(event, ProviderTextDelta):
                        if thinking_active:
                            thinking_active = False
                            yield ThinkingFinished()
                        yield TextDelta(text=event.text)
                    elif isinstance(event, ProviderResponseFinished):
                        response = event.exchange
            except asyncio.CancelledError:
                await self._sessions.append("turn_cancelled", {"phase": "provider_request"})
                yield AgentCancelled(message="Provider request cancelled")
                return
            except Exception as error:
                message = self._redacted_text(error)
                await self._sessions.append(
                    "turn_failed", {"phase": "provider_request", "message": message}
                )
                yield AgentFailed(message=f"Provider request failed: {message}")
                return

            if thinking_active:
                yield ThinkingFinished()
            elif response is not None and any(
                isinstance(block, (ThinkingBlock, RedactedThinkingBlock))
                for block in response.blocks
            ):
                yield ThinkingStarted()
                yield ThinkingFinished()

            if response is None:
                message = "Provider response ended without a completed exchange"
                await self._sessions.append("turn_failed", {"message": message})
                yield AgentFailed(message=message)
                return

            await self._sessions.append("assistant_exchange", response)
            if self._context_manager is not None:
                self._context_manager.record_provider_usage(
                    request_history,
                    response.usage,
                    self._supplemental_characters(system_instructions),
                )
            if response.tool_uses:
                results: list[ToolResultBlock] = []
                for call_index, call in enumerate(response.tool_uses):
                    await self._sessions.append(
                        "tool_started",
                        {
                            "call_id": call.call_id,
                            "tool_name": call.name,
                            "input": call.input,
                        },
                    )
                    yield ToolStarted(
                        call_id=call.call_id,
                        tool_name=call.name,
                        arguments=self._redacted_mapping(call.input),
                    )
                    try:
                        result = await self._dispatcher.execute(call)
                    except asyncio.CancelledError:
                        await self._sessions.append(
                            "tool_cancelled",
                            {"call_id": call.call_id, "tool_name": call.name},
                        )
                        results.append(
                            ToolResultBlock(
                                call.call_id,
                                "cancelled_by_user",
                                True,
                                {"cancelled": True},
                            )
                        )
                        for remaining in response.tool_uses[call_index + 1 :]:
                            results.append(
                                ToolResultBlock(
                                    remaining.call_id,
                                    "cancelled_before_execution",
                                    True,
                                    {"cancelled": True},
                                )
                            )
                        continuation = ToolContinuationExchange(response, tuple(results))
                        self._conversation.exchanges.append(continuation)
                        await self._sessions.append("tool_continuation", continuation)
                        await self._sessions.append("turn_cancelled", {"phase": "tool_execution"})
                        yield ToolFinished(
                            call_id=call.call_id,
                            tool_name=call.name,
                            is_error=True,
                            content="cancelled_by_user",
                            metadata={"cancelled": True},
                        )
                        yield AgentCancelled(message="Tool execution cancelled")
                        return
                    except Exception as error:
                        result = ToolResultBlock(
                            call.call_id,
                            self._redacted_text(error),
                            True,
                            {"unexpected_error": True},
                        )
                    results.append(result)
                    await self._sessions.append(
                        "tool_finished",
                        {
                            "call_id": call.call_id,
                            "tool_name": call.name,
                            "is_error": result.is_error,
                            "model_content": result.content,
                            "metadata": result.metadata,
                        },
                    )
                    yield ToolFinished(
                        call_id=call.call_id,
                        tool_name=call.name,
                        is_error=result.is_error,
                        content=self._redacted_text(result.content),
                        metadata=self._redacted_mapping(result.metadata),
                    )
                continuation = ToolContinuationExchange(
                    assistant=response,
                    results=tuple(results),
                )
                self._conversation.exchanges.append(continuation)
                await self._sessions.append("tool_continuation", continuation)
                continue

            self._conversation.exchanges.append(response)
            if response.stop_reason == "end_turn":
                await self._sessions.append("turn_completed", {})
                yield AgentCompleted(text=response.text)
                return
            message = f"Provider stopped with reason: {response.stop_reason}"
            await self._sessions.append("turn_failed", {"message": message})
            yield AgentFailed(message=message)
            return

        message = "Maximum agent steps reached"
        await self._sessions.append("turn_failed", {"message": message})
        yield AgentFailed(message=message)

    def _redacted_text(self, value: object) -> str:
        redacted = self._display_redactor(str(value))
        return redacted if isinstance(redacted, str) else "[unavailable]"

    def _redacted_mapping(self, value: object) -> dict[str, object]:
        redacted = self._display_redactor(value)
        if not isinstance(redacted, dict):
            return {}
        return {str(key): child for key, child in redacted.items()}
