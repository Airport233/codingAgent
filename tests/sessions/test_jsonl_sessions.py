from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from coding_agent.domain import (
    AssistantExchange,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
    UnknownProviderBlock,
    UserExchange,
)
from coding_agent.sessions.jsonl import (
    JsonlSessionRepository,
    Redactor,
    SessionCorruptError,
)


@pytest.fixture
def roots() -> tuple[Path, Path, Path]:
    base = Path.cwd() / ".tmp" / "sessions"
    shutil.rmtree(base, ignore_errors=True)
    data_root = base / "user-data"
    project = base / "project"
    other_project = base / "other-project"
    project.mkdir(parents=True)
    other_project.mkdir(parents=True)
    yield data_root, project, other_project
    shutil.rmtree(base, ignore_errors=True)


@pytest.mark.asyncio
async def test_jsonl_round_trip_preserves_raw_blocks_usage_and_tool_metadata(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, _ = roots
    repository = JsonlSessionRepository(data_root)
    store = await repository.create(project)
    user = UserExchange("inspect the project")
    assistant = AssistantExchange(
        blocks=(
            ThinkingBlock("reason", signature="signed", raw={"type": "thinking", "x": 1}),
            RedactedThinkingBlock("opaque", raw={"type": "redacted_thinking", "data": "opaque"}),
            UnknownProviderBlock("future", {"type": "future", "value": 2}),
            ToolUseBlock("call-1", "shell", {"command": "echo ok"}, {"type": "tool_use"}),
        ),
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 4},
    )
    continuation = ToolContinuationExchange(
        assistant=assistant,
        results=(
            ToolResultBlock(
                "call-1",
                "exit_code: 0",
                False,
                {"exit_code": 0, "duration_ms": 12, "truncated": False},
            ),
        ),
    )
    final = AssistantExchange((TextBlock("done"),), "end_turn", {"output_tokens": 1})

    await store.append("user_exchange", user)
    await store.append("assistant_exchange", assistant)
    await store.append("tool_continuation", continuation)
    await store.append("assistant_exchange", final)
    checkpoint = {
        "reason": "manual",
        "retained_from": 1,
        "before_tokens": 100,
        "after_tokens": 40,
        "summary": "Earlier conversation summary",
    }
    await store.append("compaction_completed", checkpoint)

    lines = store.events_path.read_text(encoding="utf-8").splitlines()
    decoded = [json.loads(line) for line in lines]
    assert [event["sequence"] for event in decoded] == [1, 2, 3, 4, 5, 6]
    assert all(event["schema_version"] == 1 for event in decoded)
    assert all(event["session_id"] == store.session_id for event in decoded)
    assert all(event["event_id"] for event in decoded)

    recovered = await repository.resume_latest(project)

    assert recovered is not None
    assert recovered.conversation == (user, continuation, final)
    assert recovered.warnings == ()
    assert recovered.compaction == checkpoint
    assert len(recovered.compactions) == 1
    assert recovered.compactions[0].exchange_index == 3
    assert recovered.compactions[0].payload == checkpoint


@pytest.mark.asyncio
async def test_resume_latest_is_scoped_to_the_current_project(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, other_project = roots
    repository = JsonlSessionRepository(data_root)
    old = await repository.create(project, session_id="old")
    await old.append("user_exchange", UserExchange("old"))
    other = await repository.create(other_project, session_id="other")
    await other.append("user_exchange", UserExchange("other"))
    latest = await repository.create(project, session_id="latest")
    await latest.append("user_exchange", UserExchange("latest"))

    recovered = await repository.resume_latest(project)

    assert recovered is not None
    assert recovered.store.session_id == "latest"
    assert recovered.conversation == (UserExchange("latest"),)


@pytest.mark.asyncio
async def test_sessions_can_be_listed_and_resumed_by_id(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, other_project = roots
    repository = JsonlSessionRepository(data_root)
    first = await repository.create(project, session_id="first")
    await first.append("model_changed", {"previous": None, "current": "provider/model-a"})
    await first.append("user_exchange", UserExchange("  First\nconversation   title  "))
    await first.append("assistant_exchange", AssistantExchange((TextBlock("answer"),), "end_turn"))
    await first.append(
        "compaction_completed",
        {
            "reason": "manual",
            "strategy": "deterministic",
            "retained_from": 1,
            "before_tokens": 100,
            "after_tokens": 50,
            "summary": "summary",
        },
    )
    second = await repository.create(project, session_id="second")
    await second.append("user_exchange", UserExchange("Second conversation"))
    other = await repository.create(other_project, session_id="other")
    await other.append("user_exchange", UserExchange("Must not be listed"))

    summaries = await repository.list_sessions(project)

    assert [summary.session_id for summary in summaries] == ["second", "first"]
    first_summary = summaries[1]
    assert first_summary.title == "First conversation title"
    assert first_summary.model == "provider/model-a"
    assert first_summary.exchange_count == 2
    assert first_summary.compacted is True
    recovered = await repository.resume(project, "first")
    assert recovered is not None
    assert recovered.store.session_id == "first"
    assert recovered.conversation[0] == UserExchange("  First\nconversation   title  ")
    assert await repository.resume(project, "other") is None


@pytest.mark.asyncio
async def test_starting_a_new_session_preserves_the_closed_session_file(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, _ = roots
    repository = JsonlSessionRepository(data_root)
    previous = await repository.create(project, session_id="previous")
    await previous.append("session_closed", {})

    current = await repository.create(project, session_id="current")

    assert previous.events_path.is_file()
    assert current.events_path.is_file()
    assert previous.events_path != current.events_path


@pytest.mark.asyncio
async def test_incomplete_final_json_line_is_skipped_with_a_warning(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, _ = roots
    repository = JsonlSessionRepository(data_root)
    store = await repository.create(project)
    await store.append("user_exchange", UserExchange("safe"))
    with store.events_path.open("ab") as stream:
        stream.write(b'{"schema_version":1,"partial"')

    recovered = await repository.resume_latest(project)

    assert recovered is not None
    assert recovered.conversation == (UserExchange("safe"),)
    assert recovered.warnings == ("Skipped an incomplete final session record",)


@pytest.mark.asyncio
async def test_corrupt_middle_records_and_invalid_sequences_are_rejected(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, _ = roots
    repository = JsonlSessionRepository(data_root)
    store = await repository.create(project)
    await store.append("user_exchange", UserExchange("one"))
    await store.append("user_exchange", UserExchange("two"))
    lines = store.events_path.read_text(encoding="utf-8").splitlines()
    store.events_path.write_text(
        "\n".join([lines[0], "not-json", lines[2]]) + "\n", encoding="utf-8"
    )

    with pytest.raises(SessionCorruptError, match="line 2"):
        await repository.resume_latest(project)

    store.events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    event = json.loads(lines[1])
    event["sequence"] = 99
    lines[1] = json.dumps(event)
    store.events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(SessionCorruptError, match="sequence"):
        await repository.resume_latest(project)


@pytest.mark.asyncio
async def test_resume_repairs_unfinished_tool_calls_without_reexecuting_them(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, _ = roots
    repository = JsonlSessionRepository(data_root)
    store = await repository.create(project)
    assistant = AssistantExchange(
        (ToolUseBlock("call-1", "write_file", {"path": "x"}),),
        "tool_use",
    )
    await store.append("user_exchange", UserExchange("write x"))
    await store.append("assistant_exchange", assistant)

    recovered = await repository.resume_latest(project)

    assert recovered is not None
    continuation = recovered.conversation[-1]
    assert isinstance(continuation, ToolContinuationExchange)
    assert continuation.assistant == assistant
    assert continuation.results == (
        ToolResultBlock(
            "call-1",
            "interrupted_before_result",
            True,
            {"status": "cancelled", "recovered": True},
        ),
    )
    assert (
        json.loads(store.events_path.read_text(encoding="utf-8").splitlines()[-1])["kind"]
        == "tool_continuation"
    )


@pytest.mark.asyncio
async def test_redactor_removes_known_secrets_and_auth_fields_before_disk(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, _ = roots
    repository = JsonlSessionRepository(
        data_root,
        redactor=Redactor(secrets=("real-secret", "https://internal.example/v1")),
    )
    store = await repository.create(project)

    await store.append(
        "diagnostic",
        {
            "message": "Bearer real-secret at https://internal.example/v1",
            "authorization": "Bearer real-secret",
            "nested": {"api_key": "real-secret", "safe": "keep"},
        },
    )

    persisted = store.events_path.read_text(encoding="utf-8")
    assert "real-secret" not in persisted
    assert "internal.example" not in persisted
    assert persisted.count("[REDACTED]") >= 3
    assert '"safe":"keep"' in persisted


@pytest.mark.asyncio
async def test_concurrent_appends_are_serialized_with_unique_sequences(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, _ = roots
    store = await JsonlSessionRepository(data_root).create(project)

    await asyncio.gather(*(store.append("diagnostic", {"index": index}) for index in range(20)))

    decoded = [
        json.loads(line) for line in store.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["sequence"] for event in decoded] == list(range(1, 22))
    assert {event["payload"]["index"] for event in decoded[1:]} == set(range(20))


@pytest.mark.asyncio
async def test_unknown_optional_envelope_fields_do_not_block_recovery(
    roots: tuple[Path, Path, Path],
) -> None:
    data_root, project, _ = roots
    repository = JsonlSessionRepository(data_root)
    store = await repository.create(project)
    await store.append("user_exchange", UserExchange("keep me"))
    lines = store.events_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[1])
    event["future_optional_field"] = {"enabled": True}
    lines[1] = json.dumps(event)
    store.events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    recovered = await repository.resume_latest(project)

    assert recovered is not None
    assert recovered.conversation == (UserExchange("keep me"),)
