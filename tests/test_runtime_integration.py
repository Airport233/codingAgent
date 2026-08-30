from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator, Iterator
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel
from rich.console import Console

from coding_agent.application import AgentApplication
from coding_agent.approval import ConfigurableApprovalPolicy
from coding_agent.cli import CliTransition, _status_line, run_repl, write_console
from coding_agent.domain import (
    AssistantExchange,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolUseBlock,
    UserExchange,
)
from coding_agent.events import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    ApprovalRequested,
    TextDelta,
    ThinkingDelta,
    ThinkingFinished,
    ThinkingStarted,
    ToolFinished,
    ToolStarted,
    WarningRaised,
)
from coding_agent.providers.base import (
    ProviderEvent,
    ProviderResponseFinished,
    ProviderTextDelta,
    ProviderThinkingDelta,
)
from coding_agent.providers.fake import FakeProvider
from coding_agent.runtime import RuntimeConfigurationError, RuntimeSettings, create_runtime
from coding_agent.sessions.jsonl import Redactor
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.base import ToolOutput
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tools.shell import ShellRiskVerdict


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


@pytest.mark.asyncio
async def test_new_runtime_does_not_persist_an_empty_session(tmp_path: Path) -> None:
    settings = RuntimeSettings.from_environment(
        workspace=tmp_path,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
    )

    unused = await create_runtime(settings, provider=FakeProvider([]))
    await unused.application.close_session()
    await unused.aclose()
    assert not list((tmp_path / "data").rglob("*.jsonl"))

    active = await create_runtime(
        settings,
        provider=FakeProvider([AssistantExchange((TextBlock("answer"),), "end_turn")]),
    )
    _ = [event async for event in active.application.run("first prompt")]
    await active.aclose()

    session_files = list((tmp_path / "data").rglob("*.jsonl"))
    assert len(session_files) == 1
    records = session_files[0].read_text(encoding="utf-8").splitlines()
    assert len(records) >= 4
    assert '"kind":"session_started"' in records[0]
    assert '"kind":"model_changed"' in records[1]
    assert '"kind":"user_exchange"' in records[2]


@pytest.mark.asyncio
async def test_runtime_injects_base_prompt_with_workspace_into_first_request(
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings.from_environment(
        workspace=tmp_path,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
    )
    provider = FakeProvider([AssistantExchange((TextBlock("answer"),), "end_turn")])

    runtime = await create_runtime(settings, provider=provider)
    _ = [event async for event in runtime.application.run("do something")]
    await runtime.aclose()

    assert str(tmp_path.resolve()) in provider.system_instructions[0]
    assert "relative to the working directory" in provider.system_instructions[0]


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
async def test_auto_mode_never_blocks_a_flagged_shell_command_but_warns(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    (workspace / "doomed").mkdir(parents=True)
    settings = RuntimeSettings.from_environment(
        workspace=workspace,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
        approval_mode="auto",
    )
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("call-1", "shell", {"command": "rm -rf doomed"}),), "tool_use"
            ),
            AssistantExchange((TextBlock("done"),), "end_turn"),
        ]
    )

    runtime = await create_runtime(settings, provider=provider)
    try:
        events = [event async for event in runtime.application.run("clean up")]
    finally:
        await runtime.aclose()

    assert not any(isinstance(event, ApprovalRequested) for event in events)
    assert any(isinstance(event, WarningRaised) for event in events)
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.is_error is False
    assert not (workspace / "doomed").exists()


@pytest.mark.asyncio
async def test_deny_mode_blocks_a_flagged_shell_command_without_asking_or_warning(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    (workspace / "doomed").mkdir(parents=True)
    settings = RuntimeSettings.from_environment(
        workspace=workspace,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
        approval_mode="deny",
    )
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("call-1", "shell", {"command": "rm -rf doomed"}),), "tool_use"
            ),
            AssistantExchange((TextBlock("done"),), "end_turn"),
        ]
    )

    runtime = await create_runtime(settings, provider=provider)
    try:
        events = [event async for event in runtime.application.run("clean up")]
    finally:
        await runtime.aclose()

    assert not any(isinstance(event, ApprovalRequested) for event in events)
    assert not any(isinstance(event, WarningRaised) for event in events)
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.is_error is True
    assert finished.metadata == {"denied": True}
    assert (workspace / "doomed").exists()


@pytest.mark.asyncio
async def test_configured_rule_still_denies_under_auto_mode(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    (workspace / "doomed").mkdir(parents=True)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[permissions.shell_rules]\n"rm -rf doomed" = "deny"\n', encoding="utf-8"
    )
    settings = RuntimeSettings.load(
        workspace=workspace,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
        config_path=config_path,
        approval_mode="auto",
    )
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("call-1", "shell", {"command": "rm -rf doomed"}),), "tool_use"
            ),
            AssistantExchange((TextBlock("done"),), "end_turn"),
        ]
    )

    runtime = await create_runtime(settings, provider=provider)
    try:
        events = [event async for event in runtime.application.run("clean up")]
    finally:
        await runtime.aclose()

    assert not any(isinstance(event, ApprovalRequested) for event in events)
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.is_error is True
    assert finished.metadata == {"denied": True}
    assert (workspace / "doomed").exists()


@pytest.mark.asyncio
async def test_configured_rule_still_allows_under_deny_mode(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir(parents=True)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[permissions.shell_rules]\n"echo hello" = "allow"\n', encoding="utf-8"
    )
    settings = RuntimeSettings.load(
        workspace=workspace,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
        config_path=config_path,
        approval_mode="deny",
    )
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("call-1", "shell", {"command": "echo hello"}),), "tool_use"
            ),
            AssistantExchange((TextBlock("done"),), "end_turn"),
        ]
    )

    runtime = await create_runtime(settings, provider=provider)
    try:
        events = [event async for event in runtime.application.run("say hi")]
    finally:
        await runtime.aclose()

    assert not any(isinstance(event, ApprovalRequested) for event in events)
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.is_error is False
    assert finished.metadata.get("denied") is not True


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
    assert request[0].content.startswith("<coding-agent-context-checkpoint>")
    assert "historical background, not a new user request" in request[0].content
    assert summary in request[0].content
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
    assert "Context estimate:" in "".join(output)
    assert "%" in "".join(output)
    assert "Compacted context with provider summary:" in "".join(output)


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
    assert "[tool] shell [.] $ python -m unittest" in rendered
    assert "stdout:\nran python -m unittest" in rendered
    assert "[tool] shell done" in rendered


@pytest.mark.asyncio
async def test_repl_shows_and_switches_approval_mode() -> None:
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy("auto", frozenset({"shell"})),
    )
    inputs: Iterator[str] = iter(("/mode", "/mode deny", "/exit"))
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(application, read_input=read_input, write_output=output.append)

    rendered = "".join(output)
    assert "Approval mode: auto" in rendered
    assert "Approval mode switched to deny." in rendered
    assert application.approval_mode() == "deny"


@pytest.mark.asyncio
async def test_repl_lists_skills_and_reports_usage_errors() -> None:
    from coding_agent.skills import SkillLoader

    skills = SkillLoader.default().load()
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        skills=skills,
    )
    inputs: Iterator[str] = iter(("/skills", "/skill", "/skill nonexistent do thing", "/exit"))
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(application, read_input=read_input, write_output=output.append)

    rendered = "".join(output)
    assert "Skills:" in rendered
    assert "test-fix" in rendered
    assert "Usage: /skill <name> <task>" in rendered


@pytest.mark.asyncio
async def test_repl_reports_skill_install_without_npx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _cmd: None)
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
    )
    inputs: Iterator[str] = iter(("/skill install owner/repo", "/exit"))
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(
        application,
        read_input=read_input,
        write_output=output.append,
        workspace=str(tmp_path),
    )

    rendered = "".join(output)
    assert "npx is not installed" in rendered


@pytest.mark.asyncio
async def test_close_session_is_idempotent() -> None:
    store = InMemorySessionStore()
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        store,
    )
    await application.close_session()
    assert store.kinds[-1] == "session_closed"
    await application.close_session()
    assert store.kinds[-1] == "session_closed"


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
async def test_shell_classifier_logs_a_classification_event_for_every_shell_call() -> None:
    verdict = ShellRiskVerdict(
        tier="elevated",
        matched_rule=None,
        escapes_workspace=False,
        reason="matches a built-in destructive-command pattern",
        forced_action="ask",
    )
    store = InMemorySessionStore()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (ToolUseBlock("call-1", "shell", {"command": "git push origin main"}),),
                    "tool_use",
                ),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": _RecordingShellTool()})),
        store,
        shell_classifier=lambda call: verdict if call.name == "shell" else None,
    )

    async for event in application.run("push"):
        if isinstance(event, ApprovalRequested):
            await application.resolve_approval(event.request_id, "allow_once")

    classified = next(
        record.payload for record in store.records if record.kind == "shell_command_classified"
    )
    assert classified["tier"] == "elevated"
    assert classified["reason"] == verdict.reason
    assert classified["command"] == "git push origin main"


@pytest.mark.asyncio
async def test_guardian_note_is_surfaced_and_logged_when_enabled() -> None:
    store = InMemorySessionStore()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (ToolUseBlock("call-1", "shell", {"command": "rm -rf build"}),), "tool_use"
                ),
                AssistantExchange((TextBlock("Removes the build directory."),), "end_turn"),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": _RecordingShellTool()})),
        store,
        approval_policy=ConfigurableApprovalPolicy("ask", frozenset({"shell"})),
        guardian_enabled=True,
        display_redactor=Redactor(("build",)).redact,
    )

    requested: ApprovalRequested | None = None
    async for event in application.run("clean"):
        if isinstance(event, ApprovalRequested):
            requested = event
            await application.resolve_approval(event.request_id, "allow_once")

    assert requested is not None
    assert requested.guardian_note is not None
    assert requested.guardian_note == "Removes the [REDACTED] directory."
    reviewed = next(
        record.payload for record in store.records if record.kind == "guardian_reviewed"
    )
    assert reviewed["failed"] is False
    assert reviewed["note"] == "Removes the [REDACTED] directory."


@pytest.mark.asyncio
async def test_guardian_failure_does_not_block_approval() -> None:
    class GuardianFailsOnSecondCallProvider:
        def __init__(self) -> None:
            self._calls = 0

        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            self._calls += 1
            if self._calls == 1:
                yield ProviderResponseFinished(
                    exchange=AssistantExchange(
                        (ToolUseBlock("call-1", "shell", {"command": "rm -rf build"}),),
                        "tool_use",
                    )
                )
            elif self._calls == 2:
                raise RuntimeError("guardian model unavailable")
            else:
                yield ProviderResponseFinished(
                    exchange=AssistantExchange((TextBlock("done"),), "end_turn")
                )

    store = InMemorySessionStore()
    application = AgentApplication(
        GuardianFailsOnSecondCallProvider(),
        ToolDispatcher(ToolCatalog({"shell": _RecordingShellTool()})),
        store,
        approval_policy=ConfigurableApprovalPolicy("ask", frozenset({"shell"})),
        guardian_enabled=True,
    )

    requested: ApprovalRequested | None = None
    events = []
    async for event in application.run("clean"):
        events.append(event)
        if isinstance(event, ApprovalRequested):
            requested = event
            await application.resolve_approval(event.request_id, "allow_once")

    assert requested is not None
    assert requested.guardian_note is None
    assert isinstance(events[-1], AgentCompleted)
    reviewed = next(
        record.payload for record in store.records if record.kind == "guardian_reviewed"
    )
    assert reviewed["failed"] is True
    assert reviewed["note"] is None


class _ShellCommandInput(BaseModel):
    command: str


class _RecordingShellTool:
    name = "shell"
    description = "Run a command"
    input_model = _ShellCommandInput

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        return ToolOutput("exit_code: 0")


@pytest.mark.asyncio
async def test_thinking_is_closed_when_the_provider_fails_mid_thought() -> None:
    class FailsWhileThinkingProvider:
        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            yield ProviderThinkingDelta(thinking="considering the request")
            raise RuntimeError("connection dropped")

    store = InMemorySessionStore()
    application = AgentApplication(
        FailsWhileThinkingProvider(),
        ToolDispatcher(ToolCatalog({})),
        store,
    )

    events = [event async for event in application.run("question")]

    thinking_started = [e for e in events if isinstance(e, ThinkingStarted)]
    thinking_finished = [e for e in events if isinstance(e, ThinkingFinished)]
    assert len(thinking_started) == 1
    assert len(thinking_finished) == 1
    assert events.index(thinking_finished[0]) > events.index(thinking_started[0])
    assert isinstance(events[-1], AgentFailed)
    assert events.index(thinking_finished[0]) < events.index(events[-1])


@pytest.mark.asyncio
async def test_thinking_is_closed_when_the_provider_is_cancelled_mid_thought() -> None:
    class CancelledWhileThinkingProvider:
        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            yield ProviderThinkingDelta(thinking="considering the request")
            raise asyncio.CancelledError()

    store = InMemorySessionStore()
    application = AgentApplication(
        CancelledWhileThinkingProvider(),
        ToolDispatcher(ToolCatalog({})),
        store,
    )

    events = [event async for event in application.run("question")]

    thinking_finished = [e for e in events if isinstance(e, ThinkingFinished)]
    assert len(thinking_finished) == 1
    assert isinstance(events[-1], AgentCancelled)
    assert events.index(thinking_finished[0]) < events.index(events[-1])


@pytest.mark.asyncio
async def test_recovered_thinking_without_deltas_still_surfaces_its_text() -> None:
    class DeltaLessThinkingProvider:
        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            yield ProviderResponseFinished(
                exchange=AssistantExchange(
                    blocks=(
                        ThinkingBlock(thinking="a real chain of reasoning", signature="sig"),
                        TextBlock(text="Here is my answer."),
                    ),
                    stop_reason="end_turn",
                )
            )

    store = InMemorySessionStore()
    application = AgentApplication(
        DeltaLessThinkingProvider(),
        ToolDispatcher(ToolCatalog({})),
        store,
    )

    events = [event async for event in application.run("hello")]

    thinking_deltas = [e for e in events if isinstance(e, ThinkingDelta)]
    assert any(e.text == "a real chain of reasoning" for e in thinking_deltas)
    started_index = next(i for i, e in enumerate(events) if isinstance(e, ThinkingStarted))
    finished_index = next(i for i, e in enumerate(events) if isinstance(e, ThinkingFinished))
    delta_index = next(i for i, e in enumerate(events) if isinstance(e, ThinkingDelta))
    assert started_index < delta_index < finished_index


@pytest.mark.asyncio
async def test_purely_redacted_thinking_still_closes_without_a_delta() -> None:
    class RedactedOnlyProvider:
        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            yield ProviderResponseFinished(
                exchange=AssistantExchange(
                    blocks=(
                        RedactedThinkingBlock(data="opaque"),
                        TextBlock(text="answer"),
                    ),
                    stop_reason="end_turn",
                )
            )

    store = InMemorySessionStore()
    application = AgentApplication(
        RedactedOnlyProvider(),
        ToolDispatcher(ToolCatalog({})),
        store,
    )

    events = [event async for event in application.run("hello")]

    assert not any(isinstance(e, ThinkingDelta) for e in events)
    assert any(isinstance(e, ThinkingStarted) for e in events)
    assert any(isinstance(e, ThinkingFinished) for e in events)


@pytest.mark.asyncio
async def test_live_streamed_thinking_is_not_replayed_a_second_time_after_text() -> None:
    """Regression: once thinking streamed live via deltas and closed before the
    text segment, the post-loop recovery fallback must not replay the same
    thinking again after the reply -- that produced a second 'Thinking ·
    complete' panel sandwiching the reply in the TUI."""

    class LiveThenBundledProvider:
        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            yield ProviderThinkingDelta(thinking="live chain of reasoning")
            yield ProviderTextDelta(text="Here is my answer.")
            yield ProviderResponseFinished(
                exchange=AssistantExchange(
                    blocks=(
                        ThinkingBlock(thinking="live chain of reasoning", signature="sig"),
                        TextBlock(text="Here is my answer."),
                    ),
                    stop_reason="end_turn",
                )
            )

    store = InMemorySessionStore()
    application = AgentApplication(
        LiveThenBundledProvider(),
        ToolDispatcher(ToolCatalog({})),
        store,
    )

    events = [event async for event in application.run("hello")]

    assert sum(isinstance(e, ThinkingStarted) for e in events) == 1
    assert sum(isinstance(e, ThinkingFinished) for e in events) == 1
    text_index = next(i for i, e in enumerate(events) if isinstance(e, TextDelta))
    finished_index = next(i for i, e in enumerate(events) if isinstance(e, ThinkingFinished))
    assert finished_index < text_index


@pytest.mark.asyncio
async def test_trailing_live_thinking_closes_before_a_tool_call_with_no_text() -> None:
    """Thinking that ends a step directly (tool_use follows, no text delta at
    all) must still close via the main-line 'thinking_active' check, not the
    exception handlers or the no-live-delta recovery path."""

    class ThinkThenToolProvider:
        def __init__(self) -> None:
            self._calls = 0

        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            self._calls += 1
            if self._calls == 1:
                yield ProviderThinkingDelta(thinking="deciding which tool to call")
                yield ProviderResponseFinished(
                    exchange=AssistantExchange(
                        blocks=(
                            ThinkingBlock(thinking="deciding which tool to call", signature="sig"),
                            ToolUseBlock("call-1", "noop", {}),
                        ),
                        stop_reason="tool_use",
                    )
                )
            else:
                yield ProviderTextDelta(text="done")
                yield ProviderResponseFinished(
                    exchange=AssistantExchange(
                        blocks=(TextBlock(text="done"),),
                        stop_reason="end_turn",
                    )
                )

    class NoopInput(BaseModel):
        pass

    class NoopTool:
        name = "noop"
        description = "does nothing"
        input_model = NoopInput

        async def execute(self, arguments: BaseModel) -> ToolOutput:
            del arguments
            return ToolOutput("ok", {})

    store = InMemorySessionStore()
    application = AgentApplication(
        ThinkThenToolProvider(),
        ToolDispatcher(ToolCatalog({"noop": NoopTool()})),
        store,
    )

    events = [event async for event in application.run("hello")]

    assert sum(isinstance(e, ThinkingStarted) for e in events) == 1
    assert sum(isinstance(e, ThinkingFinished) for e in events) == 1
    finished_index = next(i for i, e in enumerate(events) if isinstance(e, ThinkingFinished))
    tool_started_index = next(i for i, e in enumerate(events) if isinstance(e, ToolStarted))
    assert finished_index < tool_started_index


@pytest.mark.asyncio
async def test_text_only_response_emits_no_thinking_events() -> None:
    class TextOnlyProvider:
        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            yield ProviderTextDelta(text="just an answer")
            yield ProviderResponseFinished(
                exchange=AssistantExchange(
                    blocks=(TextBlock(text="just an answer"),),
                    stop_reason="end_turn",
                )
            )

    store = InMemorySessionStore()
    application = AgentApplication(
        TextOnlyProvider(),
        ToolDispatcher(ToolCatalog({})),
        store,
    )

    events = [event async for event in application.run("hello")]

    thinking_events = (ThinkingStarted, ThinkingDelta, ThinkingFinished)
    assert not any(isinstance(e, thinking_events) for e in events)


@pytest.mark.asyncio
async def test_empty_non_redacted_thinking_block_emits_no_thinking_events() -> None:
    """A ThinkingBlock with no text and no live deltas carries nothing worth
    showing -- unlike a RedactedThinkingBlock, it should not open a panel."""

    class EmptyThinkingProvider:
        async def stream(self, *_args: object) -> AsyncIterator[ProviderEvent]:
            yield ProviderResponseFinished(
                exchange=AssistantExchange(
                    blocks=(
                        ThinkingBlock(thinking="", signature=""),
                        TextBlock(text="answer"),
                    ),
                    stop_reason="end_turn",
                )
            )

    store = InMemorySessionStore()
    application = AgentApplication(
        EmptyThinkingProvider(),
        ToolDispatcher(ToolCatalog({})),
        store,
    )

    events = [event async for event in application.run("hello")]

    thinking_events = (ThinkingStarted, ThinkingDelta, ThinkingFinished)
    assert not any(isinstance(e, thinking_events) for e in events)


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
    assert "Commands:\n  /help" in rendered
    assert "/model [provider/model]  Show or choose a model" in rendered
    assert "/resume                  Resume a saved session" in rendered
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
