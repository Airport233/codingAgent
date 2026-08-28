from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from coding_agent.cli import run_repl, write_console
from coding_agent.domain import AssistantExchange, TextBlock, UserExchange
from coding_agent.providers.fake import FakeProvider
from coding_agent.runtime import RuntimeSettings, create_runtime


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
    assert settings.workspace == tmp_path.resolve()
    assert settings.data_root == (tmp_path / "data").resolve()
    assert "private-test-credential" not in repr(settings)
    assert "private.example" not in repr(settings)


def test_runtime_settings_load_user_provider_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[general]
default_model = "local/claude-example"

[providers.local]
base_url = "https://example.invalid/anthropic"
api_key_env = "LOCAL_PROVIDER_KEY"
models = ["claude-example"]
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
    assert settings.sdk_base_url == "https://example.invalid/anthropic/"


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
async def test_repl_accepts_multiple_turns_and_exits_without_provider_work(tmp_path: Path) -> None:
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
            AssistantExchange((TextBlock("answer one"),), stop_reason="end_turn"),
            AssistantExchange((TextBlock("answer two"),), stop_reason="end_turn"),
        ]
    )
    runtime = await create_runtime(settings, provider=provider)
    inputs: Iterator[str] = iter(("question one", "question two", "/context", "/compact", "/exit"))
    output: list[str] = []

    async def read_input() -> str:
        return next(inputs)

    await run_repl(runtime.application, read_input=read_input, write_output=output.append)
    await runtime.aclose()

    assert provider.request_count == 2
    assert "answer one" in "".join(output)
    assert "answer two" in "".join(output)
    assert "Context:" in "".join(output)
    assert "too short to compact" in "".join(output)


def test_console_output_does_not_treat_model_text_as_rich_markup() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False)

    write_console(console, "[build-system]")

    assert stream.getvalue() == "[build-system]"
