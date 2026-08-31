from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from coding_agent.application import AgentApplication
from coding_agent.domain import (
    AssistantExchange,
    ConversationExchange,
    ProviderContinuationExchange,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UserExchange,
)
from coding_agent.events import AgentCompleted, AgentFailed, ToolFinished, WarningRaised
from coding_agent.providers.anthropic_messages import encode_conversation
from coding_agent.providers.anthropic_stream import AnthropicProtocolError
from coding_agent.providers.base import ProviderEvent, ProviderResponseFinished
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


class _FailsWithInvalidJsonThenSucceeds:
    def __init__(self) -> None:
        self.requests: list[tuple[ConversationExchange, ...]] = []
        self.system_instructions: list[str] = []

    async def stream(
        self,
        conversation: tuple[ConversationExchange, ...],
        _tools: object,
        system_instructions: str | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        self.requests.append(conversation)
        self.system_instructions.append(system_instructions or "")
        if len(self.requests) == 1:
            raise AnthropicProtocolError("invalid tool input JSON", code="invalid_tool_input_json")
        yield ProviderResponseFinished(exchange=AssistantExchange((TextBlock("done"),), "end_turn"))


@pytest.mark.asyncio
async def test_retryable_protocol_error_retries_the_same_request_with_failure_guidance() -> None:
    provider = _FailsWithInvalidJsonThenSucceeds()
    store = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        store,
        response_retry_limit=2,
    )

    events = [event async for event in application.run("finish the task")]

    assert isinstance(events[-1], AgentCompleted)
    assert len(provider.requests) == 2
    assert provider.requests[0] == provider.requests[1]
    assert "invalid tool input JSON" not in provider.system_instructions[0]
    assert "invalid tool input JSON" in provider.system_instructions[1]
    assert sum(isinstance(event, WarningRaised) for event in events) == 1
    retry = next(record.payload for record in store.records if record.kind == "provider_retry")
    assert retry == {
        "reason": "invalid_tool_input_json",
        "attempt": 1,
        "max_attempts": 2,
    }
    assert store.kinds.count("assistant_exchange") == 1


@pytest.mark.asyncio
async def test_max_tokens_without_tools_continues_from_the_partial_response() -> None:
    provider = FakeProvider(
        [
            AssistantExchange((TextBlock("partial answer"),), "max_tokens"),
            AssistantExchange((TextBlock("short complete answer"),), "end_turn"),
        ]
    )
    store = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        store,
        response_retry_limit=2,
    )

    events = [event async for event in application.run("answer")]

    assert isinstance(events[-1], AgentCompleted)
    assert events[-1].text == "short complete answer"
    assert provider.request_count == 2
    assert provider.requests[0] == (UserExchange("answer"),)
    assert len(provider.requests[1]) == 2
    continuation = provider.requests[1][-1]
    assert isinstance(continuation, ProviderContinuationExchange)
    assert continuation.assistant.text == "partial answer"
    assert "do not repeat" in continuation.instruction.casefold()
    persisted = [record.payload for record in store.records if record.kind == "assistant_exchange"]
    assert [exchange.text for exchange in persisted] == ["short complete answer"]
    assert store.kinds.count("provider_continuation") == 1
    retry = next(record.payload for record in store.records if record.kind == "provider_retry")
    assert retry["reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_thinking_only_max_tokens_continues_from_the_preserved_response() -> None:
    partial = AssistantExchange(
        (
            ThinkingBlock(
                "long analysis up to the output boundary",
                signature="signed-thinking",
                raw={
                    "type": "thinking",
                    "thinking": "long analysis up to the output boundary",
                    "signature": "signed-thinking",
                },
            ),
        ),
        "max_tokens",
    )

    class _ThinkingThenCompletes:
        def __init__(self) -> None:
            self.requests: list[list[dict[str, object]]] = []

        async def stream(
            self,
            conversation: tuple[ConversationExchange, ...],
            _tools: object,
            _system_instructions: str | None = None,
        ) -> AsyncIterator[ProviderEvent]:
            self.requests.append(encode_conversation(conversation))
            response = (
                partial
                if len(self.requests) == 1
                else AssistantExchange((TextBlock("complete"),), "end_turn")
            )
            yield ProviderResponseFinished(exchange=response)

    provider = _ThinkingThenCompletes()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        response_retry_limit=2,
    )

    events = [event async for event in application.run("solve the task")]

    assert isinstance(events[-1], AgentCompleted)
    assert provider.requests[1][:2] == [
        {"role": "user", "content": "solve the task"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "long analysis up to the output boundary",
                    "signature": "signed-thinking",
                }
            ],
        },
    ]
    continuation = provider.requests[1][2]
    assert continuation["role"] == "user"
    assert "continue" in str(continuation["content"]).casefold()
    assert "do not repeat" in str(continuation["content"]).casefold()


@pytest.mark.asyncio
async def test_max_tokens_with_a_complete_tool_call_executes_it_without_retry() -> None:
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("call-1", "unknown", {}),),
                "max_tokens",
            ),
            AssistantExchange((TextBlock("done"),), "end_turn"),
        ]
    )
    store = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        store,
        response_retry_limit=2,
    )

    events = [event async for event in application.run("use a tool")]

    assert isinstance(events[-1], AgentCompleted)
    assert provider.request_count == 2
    assert any(isinstance(event, ToolFinished) for event in events)
    assert "provider_retry" not in store.kinds


@pytest.mark.asyncio
async def test_retry_limit_stops_repeated_protocol_failures() -> None:
    class _AlwaysFails:
        def __init__(self) -> None:
            self.request_count = 0

        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            self.request_count += 1
            raise AnthropicProtocolError("invalid tool input JSON", code="invalid_tool_input_json")
            yield  # pragma: no cover

    provider = _AlwaysFails()
    store = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        store,
        response_retry_limit=2,
    )

    events = [event async for event in application.run("finish")]

    assert provider.request_count == 3
    assert sum(isinstance(event, WarningRaised) for event in events) == 2
    assert isinstance(events[-1], AgentFailed)
    assert "after 2 retries" in events[-1].message
    assert store.kinds.count("provider_retry") == 2
    assert store.kinds[-1] == "turn_failed"


@pytest.mark.asyncio
async def test_retry_limit_stops_repeated_max_token_responses_and_keeps_the_last() -> None:
    provider = FakeProvider(
        [
            AssistantExchange((TextBlock("partial one"),), "max_tokens"),
            AssistantExchange((TextBlock("partial two"),), "max_tokens"),
            AssistantExchange((TextBlock("partial three"),), "max_tokens"),
        ]
    )
    store = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        store,
        response_retry_limit=2,
    )

    events = [event async for event in application.run("finish")]

    assert provider.request_count == 3
    assert sum(isinstance(event, WarningRaised) for event in events) == 2
    assert isinstance(events[-1], AgentFailed)
    assert "after 2 retries" in events[-1].message
    persisted = [record.payload for record in store.records if record.kind == "assistant_exchange"]
    assert [exchange.text for exchange in persisted] == ["partial three"]
    assert store.kinds.count("provider_retry") == 2


def test_retry_limit_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="response_retry_limit must be non-negative"):
        AgentApplication(
            FakeProvider([]),
            ToolDispatcher(ToolCatalog({})),
            InMemorySessionStore(),
            response_retry_limit=-1,
        )
