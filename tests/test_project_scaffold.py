from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


def test_secret_scanner_rejects_obvious_credentials(tmp_path: Path) -> None:
    unsafe_file = tmp_path / "unsafe.env"
    unsafe_file.write_text(
        "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789\n",
        encoding="utf-8",
    )

    result = run_python("scripts/check_secrets.py", str(unsafe_file))

    assert result.returncode == 1
    assert "possible secret" in result.stdout.lower()
    assert "sk-ant" not in result.stdout
