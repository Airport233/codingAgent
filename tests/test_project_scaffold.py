from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from coding_agent import cli
from coding_agent.cli import app
from scripts.check_secrets import contains_secret_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_RUNNER = CliRunner()


def run_python(*args: str, cwd: Path = REPOSITORY_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
    )


def test_module_help_smoke() -> None:
    result = run_python("-m", "coding_agent", "--help")

    assert result.returncode == 0, result.stderr
    assert "coding-agent" in result.stdout


def test_cli_help_via_application_entrypoint() -> None:
    result = CLI_RUNNER.invoke(app, ["--help"], prog_name="coding-agent")

    assert result.exit_code == 0
    assert "coding-agent" in result.stdout


def test_console_script_uses_stable_program_name(monkeypatch) -> None:
    calls: list[str] = []

    def fake_app(*, prog_name: str) -> None:
        calls.append(prog_name)

    monkeypatch.setattr(cli, "app", fake_app)

    cli.main()

    assert calls == ["coding-agent"]


def test_example_config_uses_only_fictional_provider_data() -> None:
    config_path = REPOSITORY_ROOT / "config.example.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    provider = config["providers"]["example"]
    assert provider["base_url"].startswith("https://example.invalid/")
    assert provider["api_key_env"] == "CODING_AGENT_EXAMPLE_API_KEY"
    assert "api_key" not in provider


def test_secret_scanner_accepts_repository_examples() -> None:
    result = run_python("scripts/check_secrets.py", "config.example.toml")

    assert result.returncode == 0, result.stdout + result.stderr


def test_secret_scanner_recognizes_obvious_credentials_without_echoing_them() -> None:
    credential = "sk-" + "ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"

    assert contains_secret_text(f"ANTHROPIC_API_KEY={credential}")
