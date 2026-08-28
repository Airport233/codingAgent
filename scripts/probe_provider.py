from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from anthropic import AsyncAnthropic

from coding_agent.domain import (
    AssistantExchange,
    RedactedThinkingBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    UserExchange,
)
from coding_agent.providers.anthropic import AnthropicMessagesProvider
from coding_agent.providers.base import (
    ProviderResponseFinished,
    ProviderTextDelta,
    ProviderUsageUpdated,
)
from coding_agent.providers.config import normalize_sdk_base_url
from coding_agent.tools.base import ToolSpec

ProbeStatus = Literal["pass", "skip", "fail"]
ThinkingMode = Literal["skip", "enabled", "effort"]
CaseCallable = Callable[[], Awaitable[str | None]]

_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_REDACTED_URL_TAIL = re.compile(r"\[REDACTED\](?:/[^\s]*)+")


class ProbeFailure(RuntimeError):
    """A Provider behavior did not satisfy the probe contract."""


@dataclass(frozen=True, slots=True)
class ProbeSettings:
    model: str
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    timeout_seconds: float = 30.0
    max_tokens: int = 512
    thinking_mode: ThinkingMode = "skip"
    thinking_budget: int = 1024
    thinking_effort: str = "high"
    probe_cancellation: bool = True

    @classmethod
    def from_environment(
        cls,
        *,
        model: str | None,
        base_url_env: str,
        api_key_env: str,
        environ: Mapping[str, str] | None = None,
        **overrides: object,
    ) -> ProbeSettings:
        values = os.environ if environ is None else environ
        base_url = values.get(base_url_env, "").strip()
        credential = values.get(api_key_env, "").strip()
        selected_model = (model or values.get("CODING_AGENT_MODEL", "")).strip()
        if not base_url:
            raise ValueError(f"Missing environment variable: {base_url_env}")
        if not credential:
            raise ValueError(f"Missing environment variable: {api_key_env}")
        if not selected_model:
            raise ValueError("Missing model: pass --model or set CODING_AGENT_MODEL")
        normalize_sdk_base_url(base_url)
        return cls(
            model=selected_model,
            base_url=base_url,
            api_key=credential,
            **overrides,  # type: ignore[arg-type]
        )

    @property
    def sdk_base_url(self) -> str:
        return normalize_sdk_base_url(self.base_url)


@dataclass(frozen=True, slots=True)
class ProbeCase:
    name: str
    run: CaseCallable


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    status: ProbeStatus
    diagnostic: str


@dataclass(frozen=True, slots=True)
class ProbeReport:
    results: tuple[ProbeResult, ...]

    @property
    def exit_code(self) -> int:
        return int(any(result.status == "fail" for result in self.results))

    def render(self) -> str:
        return "\n".join(
            f"[{result.status.upper()}] {result.name}: {result.diagnostic}"
            for result in self.results
        )


def sanitize_diagnostic(message: str, *, secrets: Sequence[str]) -> str:
    sanitized = message
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = _REDACTED_URL_TAIL.sub("[REDACTED]", sanitized)
    sanitized = _BEARER.sub(r"\1[REDACTED]", sanitized)
    return " ".join(sanitized.split())[:500]


async def run_cases(cases: Sequence[ProbeCase], *, secrets: Sequence[str]) -> ProbeReport:
    results: list[ProbeResult] = []
    for case in cases:
        try:
            diagnostic = await case.run()
        except Exception as error:
            results.append(
                ProbeResult(
                    case.name,
                    "fail",
                    sanitize_diagnostic(str(error) or type(error).__name__, secrets=secrets),
                )
            )
        else:
            status: ProbeStatus = "skip" if diagnostic is None else "pass"
            results.append(ProbeResult(case.name, status, diagnostic or "not requested"))
    return ProbeReport(tuple(results))


@dataclass(frozen=True, slots=True)
class _ObservedResponse:
    exchange: AssistantExchange
    saw_text_delta: bool
    saw_usage_event: bool


class LiveProviderProbe:
    _ECHO_TOOL = ToolSpec(
        name="probe_echo",
        description="Return a supplied probe value.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    _SECOND_TOOL = ToolSpec(
        name="probe_second",
        description="Record a second supplied probe value.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    def __init__(self, client: AsyncAnthropic, settings: ProbeSettings) -> None:
        self._client = client
        self._settings = settings
        self._seed_exchange: AssistantExchange | None = None

    def cases(self) -> tuple[ProbeCase, ...]:
        cases = [
            ProbeCase("text_stream", self.probe_text_stream),
            ProbeCase("thinking", self.probe_thinking),
            ProbeCase("thinking_tool_round_trip", self.probe_thinking_tool_round_trip),
            ProbeCase("single_tool_round_trip", self.probe_single_tool_round_trip),
            ProbeCase("tool_error_result", self.probe_tool_error_result),
            ProbeCase("multiple_tool_calls", self.probe_multiple_tool_calls),
        ]
        if self._settings.probe_cancellation:
            cases.append(ProbeCase("stream_cancellation", self.probe_stream_cancellation))
        return tuple(cases)

    async def probe_text_stream(self) -> str:
        observed = await self._request(
            (UserExchange("Reply with exactly PROBE_OK."),),
            (),
        )
        if observed.exchange.stop_reason != "end_turn":
            raise ProbeFailure(f"unexpected stop reason: {observed.exchange.stop_reason}")
        if not observed.exchange.text.strip():
            raise ProbeFailure("response did not contain text")
        if not observed.saw_text_delta:
            raise ProbeFailure("response did not stream a text delta")
        if not observed.exchange.usage or not observed.saw_usage_event:
            raise ProbeFailure("response did not expose usage")
        return "stream=yes stop_reason=end_turn usage=yes"

    async def probe_thinking(self) -> str | None:
        options = self._thinking_request_options()
        if options is None:
            return None
        extra_body, max_tokens = options
        observed = await self._request(
            (UserExchange("Think briefly, then reply with exactly PROBE_OK."),),
            (),
            extra_body=extra_body,
            max_tokens=max_tokens,
        )
        thinking_blocks = self._require_thinking_blocks(observed.exchange)
        return f"mode={self._settings.thinking_mode} blocks={len(thinking_blocks)}"

    async def probe_thinking_tool_round_trip(self) -> str | None:
        options = self._thinking_request_options()
        if options is None:
            return None
        thinking_body, max_tokens = options
        user = UserExchange(
            "You must use the probe_echo tool. Think briefly, then call probe_echo exactly once "
            "with value 'probe-value'. Do not answer without using the tool. "
            "After its result, finish without another tool call."
        )
        observed = await self._request(
            (user,),
            (self._ECHO_TOOL,),
            extra_body=thinking_body,
            max_tokens=max_tokens,
        )
        self._require_thinking_blocks(observed.exchange)
        calls = observed.exchange.tool_uses
        if observed.exchange.stop_reason != "tool_use" or len(calls) != 1:
            raise ProbeFailure("thinking response did not contain exactly one tool call")
        continuation = ToolContinuationExchange(
            observed.exchange,
            (ToolResultBlock(calls[0].call_id, "probe-value", False),),
        )
        final = await self._request(
            (user, continuation),
            (self._ECHO_TOOL,),
            extra_body=thinking_body,
            max_tokens=max_tokens,
        )
        if final.exchange.stop_reason != "end_turn":
            raise ProbeFailure("thinking tool result did not complete normally")
        return "thinking_preserved=yes tool_result=yes"

    async def probe_single_tool_round_trip(self) -> str:
        assistant = await self._get_seed_tool_exchange()
        result = ToolResultBlock(
            tool_use_id=assistant.tool_uses[0].call_id,
            content="probe-value",
            is_error=False,
        )
        observed = await self._request(
            (
                UserExchange(
                    "Call probe_echo once with value 'probe-value'. "
                    "After its result, finish without another tool call."
                ),
                ToolContinuationExchange(assistant, (result,)),
            ),
            (self._ECHO_TOOL,),
        )
        if observed.exchange.stop_reason != "end_turn":
            raise ProbeFailure(
                f"tool result was not accepted; stop reason={observed.exchange.stop_reason}"
            )
        return "tool_use=yes tool_result=yes"

    async def probe_tool_error_result(self) -> str:
        assistant = await self._get_seed_tool_exchange()
        result = ToolResultBlock(
            tool_use_id=assistant.tool_uses[0].call_id,
            content="synthetic_probe_error",
            is_error=True,
        )
        observed = await self._request(
            (
                UserExchange(
                    "Call probe_echo once. If it returns an error, acknowledge it without retrying."
                ),
                ToolContinuationExchange(assistant, (result,)),
            ),
            (self._ECHO_TOOL,),
        )
        return f"is_error accepted stop_reason={observed.exchange.stop_reason}"

    async def probe_multiple_tool_calls(self) -> str:
        observed = await self._request(
            (
                UserExchange(
                    "In one response, call both probe_echo and probe_second exactly once, "
                    "using value 'probe-value' for each. Do not return ordinary text."
                ),
            ),
            (self._ECHO_TOOL, self._SECOND_TOOL),
            extra_body={"tool_choice": {"type": "any", "disable_parallel_tool_use": False}},
        )
        names = tuple(call.name for call in observed.exchange.tool_uses)
        if set(names) != {"probe_echo", "probe_second"} or len(names) != 2:
            raise ProbeFailure(f"expected two distinct tool calls, received {len(names)}")
        if len({call.call_id for call in observed.exchange.tool_uses}) != 2:
            raise ProbeFailure("multiple tool calls did not have unique IDs")
        return "calls=2 unique_ids=yes"

    async def probe_stream_cancellation(self) -> str:
        first_event = asyncio.Event()
        hold_open = asyncio.Event()

        async def consume() -> None:
            async with self._client.messages.stream(
                model=self._settings.model,
                max_tokens=max(self._settings.max_tokens, 128),
                messages=[
                    {
                        "role": "user",
                        "content": "Write a long numbered list for a cancellation probe.",
                    }
                ],
            ) as stream:
                async for _event in stream:
                    first_event.set()
                    await hold_open.wait()

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(first_event.wait(), timeout=self._settings.timeout_seconds)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return "cancelled=yes"
            raise ProbeFailure("stream task ignored cancellation")
        except TimeoutError as error:
            if task.done():
                await task
            raise ProbeFailure("stream produced no cancellable event before timeout") from error
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _get_seed_tool_exchange(self) -> AssistantExchange:
        if self._seed_exchange is not None:
            return self._seed_exchange
        observed = await self._request(
            (
                UserExchange(
                    "Call probe_echo exactly once with value 'probe-value'. "
                    "Do not return ordinary text."
                ),
            ),
            (self._ECHO_TOOL,),
            extra_body={"tool_choice": {"type": "tool", "name": "probe_echo"}},
        )
        calls = observed.exchange.tool_uses
        if observed.exchange.stop_reason != "tool_use" or len(calls) != 1:
            raise ProbeFailure("Provider did not return exactly one forced tool call")
        if calls[0].name != "probe_echo" or not calls[0].call_id:
            raise ProbeFailure("Provider returned an invalid forced tool call")
        self._seed_exchange = observed.exchange
        return observed.exchange

    def _thinking_request_options(self) -> tuple[dict[str, object], int] | None:
        if self._settings.thinking_mode == "skip":
            return None
        if self._settings.thinking_mode == "enabled":
            return (
                {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": self._settings.thinking_budget,
                    }
                },
                max(self._settings.max_tokens, self._settings.thinking_budget + 256),
            )
        return (
            {"output_config": {"effort": self._settings.thinking_effort}},
            self._settings.max_tokens,
        )

    @staticmethod
    def _require_thinking_blocks(
        exchange: AssistantExchange,
    ) -> tuple[ThinkingBlock | RedactedThinkingBlock, ...]:
        thinking_blocks = tuple(
            block
            for block in exchange.blocks
            if isinstance(block, (ThinkingBlock, RedactedThinkingBlock))
        )
        if not thinking_blocks:
            raise ProbeFailure("response did not contain a thinking block")
        if any(
            isinstance(block, ThinkingBlock) and not block.signature for block in thinking_blocks
        ):
            raise ProbeFailure("thinking block did not contain a signature")
        return thinking_blocks

    async def _request(
        self,
        conversation: tuple[UserExchange | ToolContinuationExchange, ...],
        tools: tuple[ToolSpec, ...],
        *,
        extra_body: dict[str, object] | None = None,
        max_tokens: int | None = None,
    ) -> _ObservedResponse:
        provider = AnthropicMessagesProvider(
            client=self._client,
            model=self._settings.model,
            max_tokens=max_tokens or self._settings.max_tokens,
            extra_body=extra_body,
        )
        exchange: AssistantExchange | None = None
        saw_text_delta = False
        saw_usage_event = False
        async for event in provider.stream(conversation, tools):
            if isinstance(event, ProviderTextDelta):
                saw_text_delta = True
            elif isinstance(event, ProviderUsageUpdated):
                saw_usage_event = True
            elif isinstance(event, ProviderResponseFinished):
                exchange = event.exchange
        if exchange is None:
            raise ProbeFailure("stream ended without a completed response")
        return _ObservedResponse(exchange, saw_text_delta, saw_usage_event)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicitly probe a real Anthropic Messages-compatible Provider."
    )
    parser.add_argument("--model", help="Provider model ID (or set CODING_AGENT_MODEL)")
    parser.add_argument("--base-url-env", default="CODING_AGENT_BASE_URL")
    parser.add_argument("--api-key-env", default="CODING_AGENT_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--thinking-mode",
        choices=("skip", "enabled", "effort"),
        default="skip",
    )
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--thinking-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--skip-cancellation", action="store_true")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.thinking_budget < 1024:
        raise ValueError("--thinking-budget must be at least 1024")


async def _run(settings: ProbeSettings) -> ProbeReport:
    credential = settings.api_key
    async with AsyncAnthropic(
        api_key=credential,
        base_url=settings.sdk_base_url,
        timeout=settings.timeout_seconds,
        max_retries=0,
    ) as client:
        probe = LiveProviderProbe(client, settings)
        return await run_cases(
            probe.cases(),
            secrets=(settings.api_key, settings.base_url, settings.sdk_base_url),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_arguments(args)
        settings = ProbeSettings.from_environment(
            model=args.model,
            base_url_env=args.base_url_env,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
            thinking_mode=args.thinking_mode,
            thinking_budget=args.thinking_budget,
            thinking_effort=args.thinking_effort,
            probe_cancellation=not args.skip_cancellation,
        )
    except ValueError as error:
        parser.error(str(error))
    report = asyncio.run(_run(settings))
    print(f"Provider probe model: {settings.model}")
    print(report.render())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
