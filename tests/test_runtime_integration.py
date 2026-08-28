from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator, Iterator
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel
from rich.console import Console

from coding_agent.application import AgentApplication
from coding_agent.cli import CliTransition, _status_line, run_repl, write_console
from coding_agent.domain import (
    AssistantExchange,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolUseBlock,
    UserExchange,
)
from coding_agent.events import AgentFailed, WarningRaised
from coding_agent.providers.base import ProviderEvent
from coding_agent.providers.fake import FakeProvider
from coding_agent.runtime import RuntimeConfigurationError, RuntimeSettings, create_runtime
from coding_agent.sessions.jsonl import Redactor
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.base import ToolOutput
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


def test_runtime_settings_use_environment_without_exposing_private_values(tmp_path: Path) -> None:
    settings = RuntimeSettings.from_environment(
        workspace=tmp_path,
        model=None,
        environ={
            "CODING_AGENT_BASE_URL": "https://private.example/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
            "CODING_AGENT_MODEL": "claude-example",
        },
        data_root=tmp_path / "data",
    )

    assert settings.model == "claude-example"
    assert settings.model_key == "default/claude-example"
    assert settings.workspace == tmp_path.resolve()
    assert settings.data_root == (tmp_path / "data").resolve()
    assert "private-test-credential" not in repr(settings)
    assert "private.example" not in repr(settings)

    application = AgentApplication(
        FakeProvider([]), ToolDispatcher(ToolCatalog({})), InMemorySessionStore()
    )
    status_line = _status_line(settings, application, "session-id")
    assert "provider=default" in status_line
    assert "model=claude-example" in status_line
    assert f"workspace={tmp_path.resolve()}" in status_line
    assert "context=unavailable" in status_line


def test_runtime_settings_load_user_provider_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[general]
default_model = "local/claude-example"

[providers.local]
base_url = "https://example.invalid/anthropic"
api_key_env = "LOCAL_PROVIDER_KEY"

[providers.local.models.claude-example]
context_window = 120000
max_output_tokens = 6000
thinking_mode = "effort"
thinking_effort = "high"

[providers.remote]
base_url = "https://remote.example.invalid/anthropic"
api_key_env = "REMOTE_PROVIDER_KEY"
auth_mode = "bearer"

[providers.remote.models.other-model]
context_window = 64000
max_output_tokens = 2000
""".strip(),
        encoding="utf-8",
    )

    settings = RuntimeSettings.load(
        workspace=tmp_path,
        model=None,
        environ={"LOCAL_PROVIDER_KEY": "private-test-credential"},
        data_root=tmp_path / "data",
        config_path=config_path,
    )

    assert settings.model == "claude-example"
    assert settings.model_key == "local/claude-example"
    assert settings.sdk_base_url == "https://example.invalid/anthropic/"
    assert settings.context_window == 120_000
    assert settings.max_tokens == 6_000
    assert settings.provider_extra_body == {"output_config": {"effort": "high"}}
    assert settings.available_models == ("local/claude-example", "remote/other-model")

    overridden = RuntimeSettings.load(
        workspace=tmp_path,
        model=None,
        environ={"LOCAL_PROVIDER_KEY": "private-test-credential"},
        data_root=tmp_path / "data",
        config_path=config_path,
        context_window=64_000,
        max_tokens=2_000,
    )
    assert overridden.context_window == 64_000
    assert overridden.max_tokens == 2_000

    switched = RuntimeSettings.load(
        workspace=tmp_path,
        model="remote/other-model",
        environ={"REMOTE_PROVIDER_KEY": "second-private-test-credential"},
        data_root=tmp_path / "data",
        config_path=config_path,
    )
    assert switched.model == "other-model"
    assert switched.model_key == "remote/other-model"
    assert switched.sdk_base_url == "https://remote.example.invalid/anthropic/"
    assert switched.context_window == 64_000
    assert switched.auth_mode == "bearer"


@pytest.mark.asyncio
async def test_runtime_resume_installs_durable_conversation_before_next_request(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "CODING_AGENT.md").write_text("Keep changes focused.", encoding="utf-8")
    settings = RuntimeSettings.from_environment(
        workspace=workspace,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
    )
    first_provider = FakeProvider(
        [AssistantExchange((TextBlock("first answer"),), stop_reason="end_turn")]
    )
    first_runtime = await create_runtime(settings, provider=first_provider)
    _ = [event async for event in first_runtime.application.run("first question")]
    await first_runtime.aclose()

    resumed_provider = FakeProvider(
        [AssistantExchange((TextBlock("second answer"),), stop_reason="end_turn")]
    )
    resumed_runtime = await create_runtime(settings, resume=True, provider=resumed_provider)
    _ = [event async for event in resumed_runtime.application.run("second question")]
    await resumed_runtime.aclose()

    request = resumed_provider.requests[0]
    assert request[0] == UserExchange("first question")
    assert request[1].text == "first answer"
    assert request[2] == UserExchange("second question")
    assert resumed_provider.system_instructions[0].endswith("Keep changes focused.")


@pytest.mark.asyncio
async def test_explicit_workspace_limits_tools_inside_a_parent_git_repository(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "outside.txt").write_text("must stay hidden", encoding="utf-8")
    workspace = tmp_path / "nested"
    workspace.mkdir()
    (workspace / "inside.txt").write_text("visible", encoding="utf-8")
    settings = RuntimeSettings.from_environment(
        workspace=workspace,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
    )
    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ToolUseBlock("call-1", "read_file", {"path": "inside.txt"}),
                    ToolUseBlock("call-2", "read_file", {"path": "outside.txt"}),
                ),
                "tool_use",
            ),
            AssistantExchange((TextBlock("done"),), "end_turn"),
        ]
    )

    runtime = await create_runtime(settings, provider=provider)
    _ = [event async for event in runtime.application.run("read the workspace file")]
    await runtime.aclose()

    continuation = provider.requests[1][-1]
    assert isinstance(continuation, ToolContinuationExchange)
    assert continuation.results[0].content == "1: visible"
    assert continuation.results[1].is_error is True
    assert "does not exist" in continuation.results[1].content
    normalized_workspace = str(workspace.resolve())
    if os.name == "nt":
        normalized_workspace = normalized_workspace.casefold()
    expected_project_key = hashlib.sha256(normalized_workspace.encode()).hexdigest()[:24]
    session_file = next((tmp_path / "data" / "sessions").rglob("*.jsonl"))
    assert session_file.parent.name == expected_project_key


@pytest.mark.asyncio
async def test_runtime_resume_with_new_model_excludes_old_thinking_from_requests(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    environment = {
        "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
        "CODING_AGENT_API_KEY": "private-test-credential",
    }
    first_settings = RuntimeSettings.from_environment(
        workspace=workspace,
        model="first/shared-model",
        environ=environment,
        data_root=tmp_path / "data",
    )
    first_provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ThinkingBlock("private reasoning", signature="signed"),
                    TextBlock("public answer"),
                ),
                stop_reason="end_turn",
            )
        ]
    )
    first_runtime = await create_runtime(first_settings, provider=first_provider)
    _ = [event async for event in first_runtime.application.run("first question")]
    await first_runtime.aclose()

    second_settings = RuntimeSettings.from_environment(
        workspace=workspace,
        model="second/shared-model",
        environ=environment,
        data_root=tmp_path / "data",
    )
    resumed_provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ThinkingBlock("new model reasoning", signature="new-signed"),
                    TextBlock("second answer"),
                ),
                stop_reason="end_turn",
            )
        ]
    )
    resumed_runtime = await create_runtime(second_settings, resume=True, provider=resumed_provider)
    resumed_events = [event async for event in resumed_runtime.application.run("second question")]
    await resumed_runtime.aclose()

    prior_assistant = resumed_provider.requests[0][1]
    assert any(isinstance(event, WarningRaised) for event in resumed_events)
    assert isinstance(prior_assistant, AssistantExchange)
    assert prior_assistant.blocks == (TextBlock("public answer"),)

    third_provider = FakeProvider(
        [AssistantExchange((TextBlock("third answer"),), stop_reason="end_turn")]
    )
    third_runtime = await create_runtime(second_settings, resume=True, provider=third_provider)
    _ = [event async for event in third_runtime.application.run("third question")]
    await third_runtime.aclose()

    old_assistant = third_provider.requests[0][1]
    new_assistant = third_provider.requests[0][3]
    assert isinstance(old_assistant, AssistantExchange)
    assert old_assistant.blocks == (TextBlock("public answer"),)
    assert isinstance(new_assistant, AssistantExchange)
    assert new_assistant.blocks[0] == ThinkingBlock("new model reasoning", signature="new-signed")
    session_text = next((tmp_path / "data" / "sessions").rglob("*.jsonl")).read_text()
    assert "private reasoning" in session_text
    assert '"kind":"model_changed"' in session_text


@pytest.mark.asyncio
async def test_runtime_resume_restores_the_latest_compaction_checkpoint(tmp_path: Path) -> None:
    settings = RuntimeSettings.from_environment(
        workspace=tmp_path,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
    )
    summary = (
        "task_goal: preserve the active task\n"
        "user_constraints: keep history durable\n"
        "decisions: compact old exchanges\n"
        "files_read: none\n"
        "files_modified: none\n"
        "commands_and_results: none\n"
        "verification_status: pending\n"
        "known_failures: none\n"
        "pending_work: continue"
    )
    first_provider = FakeProvider(
        [
            *(
                AssistantExchange((TextBlock(f"long answer {index} " * 50),), "end_turn")
                for index in range(4)
            ),
            AssistantExchange((TextBlock(summary),), "end_turn"),
        ]
    )
    runtime = await create_runtime(settings, provider=first_provider)
    for index in range(4):
        _ = [event async for event in runtime.application.run(f"long question {index} " * 50)]
    checkpoint = await runtime.application.compact_context()
    await runtime.aclose()
    assert checkpoint is not None

    resumed_provider = FakeProvider([AssistantExchange((TextBlock("resumed answer"),), "end_turn")])
    resumed = await create_runtime(settings, resume=True, provider=resumed_provider)
    _ = [event async for event in resumed.application.run("continue")]
    await resumed.aclose()

    request = resumed_provider.requests[0]
    assert isinstance(request[0], UserExchange)
    assert request[0].content == summary
    assert all(
        not isinstance(exchange, UserExchange) or "long question 0" not in exchange.content
        for exchange in request
    )


@pytest.mark.asyncio
async def test_repl_accepts_multiple_turns_and_compacts_context(tmp_path: Path) -> None:
    settings = RuntimeSettings.from_environment(
        workspace=tmp_path,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
    )
    provider = FakeProvider(
        [
            AssistantExchange((TextBlock("answer one " * 80),), stop_reason="end_turn"),
            AssistantExchange((TextBlock("answer two " * 80),), stop_reason="end_turn"),
            AssistantExchange((TextBlock("answer three " * 80),), stop_reason="end_turn"),
            AssistantExchange((TextBlock("answer four " * 80),), stop_reason="end_turn"),
            AssistantExchange(
                (
                    TextBlock(
                        "task_goal: answer the questions\n"
                        "user_constraints: none\n"
                        "decisions: four answers supplied\n"
                        "files_read: none\n"
                        "files_modified: none\n"
                        "commands_and_results: none\n"
                        "verification_status: not applicable\n"
                        "known_failures: none\n"
                        "pending_work: continue the conversation"
                    ),
                ),
                stop_reason="end_turn",
            ),
        ]
    )
    runtime = await create_runtime(settings, provider=provider)
    inputs: Iterator[str] = iter(
        (
            "question one " * 80,
            "question two " * 80,
            "question three " * 80,
            "question four " * 80,
            "/context",
            "/compact",
            "/exit",
        )
    )
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(runtime.application, read_input=read_input, write_output=output.append)
    await runtime.aclose()

    assert provider.request_count == 5
    assert "answer one" in "".join(output)
    assert "answer four" in "".join(output)
    assert "Context:" in "".join(output)
    assert "%" in "".join(output)
    assert "Compacted context:" in "".join(output)


@pytest.mark.asyncio
async def test_repl_shows_shell_details_and_toggles_thinking() -> None:
    class ShellInput(BaseModel):
        command: str

    class VisibleShell:
        name = "shell"
        description = "Test shell"
        input_model = ShellInput

        async def execute(self, arguments: BaseModel) -> ToolOutput:
            parsed = ShellInput.model_validate(arguments)
            return ToolOutput(f"stdout:\nran {parsed.command}", {"exit_code": 0})

    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ThinkingBlock("inspect carefully", signature="must-not-be-rendered"),
                    ToolUseBlock("call-1", "shell", {"command": "python -m unittest"}),
                ),
                "tool_use",
            ),
            AssistantExchange((TextBlock("finished"),), "end_turn"),
        ]
    )
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({"shell": VisibleShell()})),
        InMemorySessionStore(),
    )
    inputs: Iterator[str] = iter(("/thinking", "run tests", "/exit"))
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(application, read_input=read_input, write_output=output.append)

    rendered = "".join(output)
    assert "Thinking details: shown." in rendered
    assert "[thinking] inspect carefully" in rendered
    assert "must-not-be-rendered" not in rendered
    assert "[tool] shell $ python -m unittest" in rendered
    assert "stdout:\nran python -m unittest" in rendered
    assert "[tool] shell done" in rendered


@pytest.mark.asyncio
async def test_repl_hides_thinking_content_by_default() -> None:
    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ThinkingBlock("private chain", signature="private-signature"),
                    TextBlock("public answer"),
                ),
                "end_turn",
            )
        ]
    )
    store = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        store,
    )
    inputs: Iterator[str] = iter(("question", "/exit"))
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(application, read_input=read_input, write_output=output.append)

    rendered = "".join(output)
    assert "[thinking] working..." in rendered
    assert "[thinking] done" in rendered
    assert "private chain" not in rendered
    assert "private-signature" not in rendered
    assert "public answer" in rendered
    persisted = next(
        record.payload for record in store.records if record.kind == "assistant_exchange"
    )
    assert isinstance(persisted, AssistantExchange)
    assert persisted.blocks[0] == ThinkingBlock("private chain", signature="private-signature")


@pytest.mark.asyncio
async def test_provider_errors_are_redacted_before_reaching_cli_events() -> None:
    class FailingProvider:
        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            raise RuntimeError("request failed with secret-value")
            yield  # pragma: no cover

    store = InMemorySessionStore()
    application = AgentApplication(
        FailingProvider(),
        ToolDispatcher(ToolCatalog({})),
        store,
        display_redactor=Redactor(("secret-value",)).redact,
    )

    events = [event async for event in application.run("question")]

    assert isinstance(events[-1], AgentFailed)
    assert "[REDACTED]" in events[-1].message
    assert "secret-value" not in events[-1].message
    assert store.kinds[-1] == "turn_failed"


@pytest.mark.asyncio
async def test_repl_lists_and_switches_models_then_starts_a_new_session() -> None:
    initial_store = InMemorySessionStore()
    switched_store = InMemorySessionStore()
    cleared_store = InMemorySessionStore()
    initial = AgentApplication(FakeProvider([]), ToolDispatcher(ToolCatalog({})), initial_store)
    switched = AgentApplication(FakeProvider([]), ToolDispatcher(ToolCatalog({})), switched_store)
    cleared = AgentApplication(FakeProvider([]), ToolDispatcher(ToolCatalog({})), cleared_store)
    selected_models: list[str] = []

    async def switch_model(target: str | None) -> CliTransition:
        assert target is not None
        selected_models.append(target)
        return CliTransition(switched, target, "same-session", ("one/model", "two/model"))

    async def clear_session(_unused: str | None) -> CliTransition:
        await switched.close_session()
        return CliTransition(cleared, "two/model", "new-session", ("one/model", "two/model"))

    inputs: Iterator[str] = iter(("/model", "/model two/model", "/clear", "/exit"))
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(
        initial,
        read_input=read_input,
        write_output=output.append,
        model="one/model",
        session_id="same-session",
        available_models=("one/model", "two/model"),
        switch_model=switch_model,
        clear_session=clear_session,
    )

    rendered = "".join(output)
    assert "Model: one/model; available: one/model, two/model" in rendered
    assert "Model switched to two/model; session=same-session." in rendered
    assert "Started a new empty session: new-session." in rendered
    assert selected_models == ["two/model"]
    assert switched_store.kinds == ["session_closed"]
    assert cleared_store.kinds == ["session_closed"]


@pytest.mark.asyncio
async def test_repl_reports_unavailable_commands_and_toggles_thinking_off() -> None:
    application = AgentApplication(
        FakeProvider([]), ToolDispatcher(ToolCatalog({})), InMemorySessionStore()
    )
    inputs: Iterator[str] = iter(
        (
            " ",
            "/help",
            "/model",
            "/model unavailable",
            "/clear",
            "/thinking",
            "/thinking",
            "/context",
            "/compact",
            "/unknown argument",
            "/exit",
        )
    )
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(
        application,
        read_input=read_input,
        write_output=output.append,
        model="current/model",
    )

    rendered = "".join(output)
    assert "Commands: /help" in rendered
    assert "Model: current/model; available: current/model" in rendered
    assert "Model switching is unavailable." in rendered
    assert "Starting a new session is unavailable." in rendered
    assert "Thinking details: shown." in rendered
    assert "Thinking details: hidden." in rendered
    assert "Context management is unavailable." in rendered
    assert "Context is too short to compact." in rendered
    assert "Unknown command: /unknown. Use /help." in rendered


@pytest.mark.asyncio
async def test_repl_keeps_running_after_transition_errors() -> None:
    application = AgentApplication(
        FakeProvider([]), ToolDispatcher(ToolCatalog({})), InMemorySessionStore()
    )

    async def switch_model(target: str | None) -> CliTransition:
        if target == "invalid/model":
            raise RuntimeConfigurationError("not configured")
        raise RuntimeError("unexpected internal detail")

    async def clear_session(_unused: str | None) -> CliTransition:
        raise RuntimeError("unexpected internal detail")

    inputs: Iterator[str] = iter(("/model invalid/model", "/model broken/model", "/clear", "/exit"))
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(
        application,
        read_input=read_input,
        write_output=output.append,
        switch_model=switch_model,
        clear_session=clear_session,
    )

    rendered = "".join(output)
    assert "Model switch failed: not configured" in rendered
    assert "[error] Model switch failed." in rendered
    assert "[error] Unable to start a new session." in rendered
    assert "unexpected internal detail" not in rendered


def test_console_output_does_not_treat_model_text_as_rich_markup() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False)

    write_console(console, "[build-system]")

    assert stream.getvalue() == "[build-system]"
