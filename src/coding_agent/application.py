from __future__ import annotations

from collections.abc import AsyncIterator

from coding_agent.domain import (
    AssistantExchange,
    Conversation,
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
from coding_agent.providers.base import Provider, ProviderResponseFinished, ProviderTextDelta
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.dispatcher import ToolDispatcher


class AgentApplication:
    def __init__(
        self,
        provider: Provider,
        dispatcher: ToolDispatcher,
        sessions: InMemorySessionStore,
        max_steps: int = 20,
    ) -> None:
        self._provider = provider
        self._dispatcher = dispatcher
        self._sessions = sessions
        self._max_steps = max_steps
        self._conversation = Conversation()

    async def run(self, prompt: str) -> AsyncIterator[CoreEvent]:
        user_exchange = UserExchange(content=prompt)
        self._conversation.exchanges.append(user_exchange)
        await self._sessions.append("user_exchange", user_exchange)
        yield AgentStarted(prompt=prompt)

        for _step in range(self._max_steps):
            response: AssistantExchange | None = None
            async for event in self._provider.stream(
                self._conversation.snapshot(), self._dispatcher.catalog.specs
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
                    yield ToolStarted(call_id=call.call_id, tool_name=call.name)
                    result = await self._dispatcher.execute(call)
                    results.append(result)
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
