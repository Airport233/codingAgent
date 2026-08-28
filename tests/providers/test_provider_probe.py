from __future__ import annotations

import asyncio

import pytest

from scripts.probe_provider import (
    ProbeCase,
    ProbeSettings,
    normalize_sdk_base_url,
    run_cases,
    sanitize_diagnostic,
)


def test_probe_settings_load_named_environment_without_exposing_secrets() -> None:
    settings = ProbeSettings.from_environment(
        model="claude-example",
        base_url_env="TEST_BASE_URL",
        api_key_env="TEST_API_KEY",
        environ={
            "TEST_BASE_URL": "https://gateway.example.invalid/anthropic/v1/messages",
            "TEST_API_KEY": "unit-test-secret-value",
        },
    )

    assert settings.sdk_base_url == "https://gateway.example.invalid/anthropic/"
    assert settings.model == "claude-example"
    assert "unit-test-secret-value" not in repr(settings)
    assert "gateway.example.invalid" not in repr(settings)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("https://example.invalid/anthropic", "https://example.invalid/anthropic/"),
        ("https://example.invalid/anthropic/", "https://example.invalid/anthropic/"),
        ("https://example.invalid/anthropic/v1", "https://example.invalid/anthropic/"),
        (
            "https://example.invalid/anthropic/v1/messages",
            "https://example.invalid/anthropic/",
        ),
    ],
)
def test_normalize_sdk_base_url_accepts_base_or_messages_endpoint(
    configured: str, expected: str
) -> None:
    assert normalize_sdk_base_url(configured) == expected


def test_probe_settings_reject_missing_or_non_http_configuration() -> None:
    with pytest.raises(ValueError, match="TEST_API_KEY"):
        ProbeSettings.from_environment(
            model="claude-example",
            base_url_env="TEST_BASE_URL",
            api_key_env="TEST_API_KEY",
            environ={"TEST_BASE_URL": "https://example.invalid/anthropic"},
        )

    with pytest.raises(ValueError, match="HTTP"):
        ProbeSettings.from_environment(
            model="claude-example",
            base_url_env="TEST_BASE_URL",
            api_key_env="TEST_API_KEY",
            environ={
                "TEST_BASE_URL": "file:///private/provider",
                "TEST_API_KEY": "unit-test-secret-value",
            },
        )


def test_diagnostics_redact_credentials_endpoint_and_authorization_values() -> None:
    diagnostic = sanitize_diagnostic(
        "request to https://private.example/v1 failed with Bearer abc.def and secret-value",
        secrets=("https://private.example", "secret-value"),
    )

    assert diagnostic == "request to [REDACTED] failed with Bearer [REDACTED] and [REDACTED]"


@pytest.mark.asyncio
async def test_run_cases_isolates_failures_and_renders_only_sanitized_diagnostics() -> None:
    async def passing() -> str:
        return "stop_reason=end_turn usage=yes"

    async def skipped() -> str | None:
        return None

    async def failing() -> str:
        await asyncio.sleep(0)
        raise RuntimeError("secret-value rejected by https://private.example/v1")

    report = await run_cases(
        (
            ProbeCase("text_stream", passing),
            ProbeCase("thinking", skipped),
            ProbeCase("tool_call", failing),
        ),
        secrets=("secret-value", "https://private.example"),
    )

    assert [result.status for result in report.results] == ["pass", "skip", "fail"]
    assert report.exit_code == 1
    rendered = report.render()
    assert "secret-value" not in rendered
    assert "private.example" not in rendered
    assert "[PASS] text_stream" in rendered
    assert "[SKIP] thinking" in rendered
    assert "[FAIL] tool_call" in rendered
