from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

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
    ) -> None:
        self._provider = provider
        self._dispatcher = dispatcher
        self._sessions = sessions
        self._max_steps = max_steps
        self._memory_loader = memory_loader
        self._last_memory_digest: str | None = None
        self._conversation = Conversation(list(initial_exchanges))

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
            async for event in self._provider.stream(
                self._conversation.snapshot(),
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
