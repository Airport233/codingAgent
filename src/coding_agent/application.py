from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from coding_agent.context import CompactionCheckpoint, ContextManager, ContextStatus
from coding_agent.domain import (
    AssistantExchange,
    Conversation,
    ConversationExchange,
    ToolContinuationExchange,
    UserExchange,
)
from coding_agent.events import (
    AgentCompleted,
    AgentFailed,
    AgentStarted,
    CoreEvent,
    TextDelta,
    ToolFinished,
    ToolStarted,
    WarningRaised,
)
from coding_agent.memory.loader import ProjectMemoryLoader
from coding_agent.providers.base import Provider, ProviderResponseFinished, ProviderTextDelta
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
    ) -> None:
        self._provider = provider
        self._dispatcher = dispatcher
        self._sessions = sessions
        self._max_steps = max_steps
        self._memory_loader = memory_loader
        self._last_memory_digest: str | None = None
        self._conversation = Conversation(list(initial_exchanges))
        self._context_manager = context_manager

    def context_status(self) -> ContextStatus | None:
        if self._context_manager is None:
            return None
        return self._context_manager.status(self._conversation.snapshot())

    async def compact_context(self, reason: str = "manual") -> CompactionCheckpoint | None:
        if self._context_manager is None:
            return None
        history = self._conversation.snapshot()
        summarize = (
            None
            if self._context_manager.status(history).level == "hard"
            else self._request_context_summary
        )
        return await self._context_manager.compact(
            history,
            reason=reason,
            persist=self._sessions.append,
            summarize=summarize,
        )

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

        for _step in range(self._max_steps):
            response: AssistantExchange | None = None
            system_instructions: str | None = None
            if self._memory_loader is not None:
                memory = self._memory_loader.load()
                system_instructions = memory.rendered or None
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
                status = self._context_manager.status(self._conversation.snapshot())
                if status.level in {"soft", "hard"}:
                    try:
                        await self.compact_context(reason="auto")
                    except Exception:
                        yield WarningRaised(
                            "Context compaction failed; continuing with the original context"
                        )
                request_history = self._context_manager.project(self._conversation.snapshot())
                if self._context_manager.status(self._conversation.snapshot()).level == "hard":
                    yield AgentFailed(message="Context remains above the safe request limit")
                    return
            else:
                request_history = self._conversation.snapshot()
            async for event in self._provider.stream(
                request_history,
                self._dispatcher.catalog.specs,
                system_instructions,
            ):
                if isinstance(event, ProviderTextDelta):
                    yield TextDelta(text=event.text)
                elif isinstance(event, ProviderResponseFinished):
                    response = event.exchange

            if response is None:
                yield AgentFailed(message="Provider response ended without a completed exchange")
                return

            await self._sessions.append("assistant_exchange", response)
            if self._context_manager is not None:
                self._context_manager.record_provider_usage(request_history, response.usage)
            if response.tool_uses:
                results = []
                for call in response.tool_uses:
                    await self._sessions.append(
                        "tool_started",
                        {
                            "call_id": call.call_id,
                            "tool_name": call.name,
                            "input": call.input,
                        },
                    )
                    yield ToolStarted(call_id=call.call_id, tool_name=call.name)
                    try:
                        result = await self._dispatcher.execute(call)
                    except asyncio.CancelledError:
                        await self._sessions.append(
                            "tool_cancelled",
                            {"call_id": call.call_id, "tool_name": call.name},
                        )
                        await self._sessions.append("turn_cancelled", {"phase": "tool_execution"})
                        raise
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
                yield AgentCompleted(text=response.text)
                return
            yield AgentFailed(message=f"Provider stopped with reason: {response.stop_reason}")
            return

        yield AgentFailed(message="Maximum agent steps reached")
