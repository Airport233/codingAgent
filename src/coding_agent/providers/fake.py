from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from coding_agent.domain import AssistantExchange, ConversationExchange, TextBlock
from coding_agent.providers.base import ProviderEvent, ProviderResponseFinished, ProviderTextDelta
from coding_agent.tools.base import ToolSpec


class FakeProvider:
    def __init__(self, responses: Sequence[AssistantExchange]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[ConversationExchange, ...]] = []
        self.system_instructions: list[str] = []

    @property
    def request_count(self) -> int:
        return len(self.requests)

    async def stream(
        self,
        conversation: tuple[ConversationExchange, ...],
        tools: tuple[ToolSpec, ...],
        system_instructions: str | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        del tools
        self.requests.append(conversation)
        self.system_instructions.append(system_instructions or "")
        if not self._responses:
            raise RuntimeError("FakeProvider has no scripted response")
        response = self._responses.pop(0)
        for block in response.blocks:
            if isinstance(block, TextBlock):
                yield ProviderTextDelta(text=block.text)
        yield ProviderResponseFinished(exchange=response)
