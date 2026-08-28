from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from coding_agent.domain import (
    AssistantExchange,
    CompactionRecord,
    ContentBlock,
    ConversationExchange,
    RedactedThinkingBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
    UnknownProviderBlock,
    UserExchange,
)

_SCHEMA_VERSION = 1
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class SessionCorruptError(Exception):
    """The durable session cannot be safely replayed."""


class Redactor:
    _SENSITIVE_KEYS = frozenset(
        {
            "api-key",
            "api_key",
            "authorization",
            "proxy-authorization",
            "token",
            "access_token",
            "x-api-key",
        }
    )
    _BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")

    def __init__(self, secrets: Sequence[str] = ()) -> None:
        self._secrets = tuple(
            sorted((secret for secret in secrets if secret), key=len, reverse=True)
        )

    def redact(self, value: object) -> object:
        if isinstance(value, str):
            redacted = value
            for secret in self._secrets:
                redacted = redacted.replace(secret, "[REDACTED]")
            return self._BEARER.sub(r"\1[REDACTED]", redacted)
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for raw_key, child in value.items():
                key = str(raw_key)
                if key.casefold() in self._SENSITIVE_KEYS:
                    result[key] = "[REDACTED]"
                else:
                    result[key] = self.redact(child)
            return result
        if isinstance(value, (list, tuple)):
            return [self.redact(child) for child in value]
        return value


@dataclass(frozen=True, slots=True)
class SessionEvent:
    schema_version: int
    event_id: str
    session_id: str
    sequence: int
    timestamp: str
    kind: str
    payload: object


@dataclass(frozen=True, slots=True)
class RecoveredSession:
    store: JsonlSessionStore
    conversation: tuple[ConversationExchange, ...]
    warnings: tuple[str, ...]
    compaction: dict[str, object] | None = None
    model: str | None = None
    conversation_models: tuple[str | None, ...] = ()
    compactions: tuple[CompactionRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    model: str | None
    exchange_count: int
    compacted: bool


class JsonlSessionStore:
    def __init__(
        self,
        events_path: Path,
        session_id: str,
        *,
        sequence: int = 0,
        redactor: Redactor | None = None,
    ) -> None:
        self.events_path = events_path
        self.session_id = session_id
        self._sequence = sequence
        self._redactor = redactor or Redactor()
        self._write_lock = asyncio.Lock()

    async def append(self, kind: str, payload: object) -> None:
        encoded_payload = _encode_payload(payload)
        redacted_payload = self._redactor.redact(encoded_payload)
        async with self._write_lock:
            sequence = self._sequence + 1
            event = {
                "schema_version": _SCHEMA_VERSION,
                "event_id": str(uuid.uuid4()),
                "session_id": self.session_id,
                "sequence": sequence,
                "timestamp": datetime.now(UTC).isoformat(),
                "kind": kind,
                "payload": redacted_payload,
            }
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._sequence = sequence


class JsonlSessionRepository:
    def __init__(self, data_root: Path, *, redactor: Redactor | None = None) -> None:
        self._sessions_root = data_root / "sessions"
        self._redactor = redactor or Redactor()

    async def create(
        self, project_root: Path, *, session_id: str | None = None
    ) -> JsonlSessionStore:
        selected_id = session_id or uuid.uuid4().hex
        if not _SESSION_ID.fullmatch(selected_id):
            raise ValueError("session_id contains unsupported characters")
        project_key = _project_key(project_root)
        path = self._sessions_root / project_key / f"{selected_id}.jsonl"
        if path.exists():
            raise FileExistsError(f"session already exists: {selected_id}")
        store = JsonlSessionStore(path, selected_id, redactor=self._redactor)
        await store.append("session_started", {"project_key": project_key})
        return store

    async def resume_latest(self, project_root: Path) -> RecoveredSession | None:
        directory = self._sessions_root / _project_key(project_root)
        if not directory.is_dir():
            return None
        candidates = tuple(directory.glob("*.jsonl"))
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
        return await self._resume_path(latest)

    async def resume(
        self, project_root: Path, session_id: str
    ) -> RecoveredSession | None:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("session_id contains unsupported characters")
        path = self._sessions_root / _project_key(project_root) / f"{session_id}.jsonl"
        if not path.is_file():
            return None
        return await self._resume_path(path)

    async def list_sessions(self, project_root: Path) -> tuple[SessionSummary, ...]:
        directory = self._sessions_root / _project_key(project_root)
        if not directory.is_dir():
            return ()
        summaries: list[SessionSummary] = []
        for path in directory.glob("*.jsonl"):
            try:
                events, _warnings = _read_events(path, repair_incomplete=False)
                if not events:
                    continue
                conversation, _, _, _, compaction, model, _ = _replay(events)
            except SessionCorruptError:
                continue
            first_prompt = next(
                (
                    " ".join(exchange.content.split())
                    for exchange in conversation
                    if isinstance(exchange, UserExchange) and exchange.content.strip()
                ),
                "Untitled session",
            )
            summaries.append(
                SessionSummary(
                    session_id=events[0].session_id,
                    title=_truncate_title(first_prompt),
                    created_at=events[0].timestamp,
                    updated_at=events[-1].timestamp,
                    model=model,
                    exchange_count=len(conversation),
                    compacted=compaction is not None,
                )
            )
        return tuple(sorted(summaries, key=lambda item: item.updated_at, reverse=True))

    async def _resume_path(self, latest: Path) -> RecoveredSession:
        events, warnings = _read_events(latest)
        if not events:
            raise SessionCorruptError("Session has no complete records")
        store = JsonlSessionStore(
            latest,
            events[0].session_id,
            sequence=events[-1].sequence,
            redactor=self._redactor,
        )
        (
            conversation,
            conversation_models,
            unfinished,
            pending_model,
            compaction,
            model,
            compactions,
        ) = _replay(events)
        if unfinished is not None:
            repair = ToolContinuationExchange(
                assistant=unfinished,
                results=tuple(
                    ToolResultBlock(
                        call.call_id,
                        "interrupted_before_result",
                        True,
                        {"status": "cancelled", "recovered": True},
                    )
                    for call in unfinished.tool_uses
                ),
            )
            await store.append("tool_continuation", repair)
            conversation.append(repair)
            conversation_models.append(pending_model)
        return RecoveredSession(
            store,
            tuple(conversation),
            tuple(warnings),
            compaction,
            model,
            tuple(conversation_models),
            tuple(compactions),
        )


def _project_key(project_root: Path) -> str:
    normalized = str(project_root.resolve())
    if os.name == "nt":
        normalized = normalized.casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _read_events(
    path: Path, *, repair_incomplete: bool = True
) -> tuple[list[SessionEvent], list[str]]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    events: list[SessionEvent] = []
    warnings: list[str] = []
    valid_bytes = 0
    for index, raw_line in enumerate(lines, start=1):
        try:
            decoded = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            is_incomplete_tail = index == len(lines) and not raw_line.endswith((b"\n", b"\r"))
            if not is_incomplete_tail:
                raise SessionCorruptError(f"Invalid JSON at line {index}") from error
            warnings.append("Skipped an incomplete final session record")
            if repair_incomplete:
                _truncate_file(path, valid_bytes)
            break
        event = _decode_event(decoded, line_number=index)
        events.append(event)
        valid_bytes += len(raw_line)
    _validate_event_sequence(events)
    return events, warnings


def _truncate_title(value: str, limit: int = 100) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _truncate_file(path: Path, length: int) -> None:
    with path.open("r+b") as stream:
        stream.truncate(length)
        stream.flush()
        os.fsync(stream.fileno())


def _decode_event(value: object, *, line_number: int) -> SessionEvent:
    if not isinstance(value, dict):
        raise SessionCorruptError(f"Session record at line {line_number} is not an object")
    try:
        schema_version = value["schema_version"]
        event_id = value["event_id"]
        session_id = value["session_id"]
        sequence = value["sequence"]
        timestamp = value["timestamp"]
        kind = value["kind"]
        payload = value["payload"]
    except KeyError as error:
        raise SessionCorruptError(
            f"Session record at line {line_number} is missing {error.args[0]}"
        ) from error
    if schema_version != _SCHEMA_VERSION:
        raise SessionCorruptError(f"Unsupported schema version at line {line_number}")
    if not all(isinstance(item, str) for item in (event_id, session_id, timestamp, kind)):
        raise SessionCorruptError(f"Invalid session record fields at line {line_number}")
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise SessionCorruptError(f"Invalid sequence at line {line_number}")
    return SessionEvent(
        schema_version,
        cast(str, event_id),
        cast(str, session_id),
        sequence,
        cast(str, timestamp),
        cast(str, kind),
        payload,
    )


def _validate_event_sequence(events: Sequence[SessionEvent]) -> None:
    if not events:
        return
    session_id = events[0].session_id
    for expected, event in enumerate(events, start=1):
        if event.session_id != session_id:
            raise SessionCorruptError("Session ID changed within the event log")
        if event.sequence != expected:
            raise SessionCorruptError(
                f"Invalid event sequence: expected {expected}, got {event.sequence}"
            )


def _replay(
    events: Sequence[SessionEvent],
) -> tuple[
    list[ConversationExchange],
    list[str | None],
    AssistantExchange | None,
    str | None,
    dict[str, object] | None,
    str | None,
    list[CompactionRecord],
]:
    conversation: list[ConversationExchange] = []
    conversation_models: list[str | None] = []
    pending: AssistantExchange | None = None
    pending_model: str | None = None
    compaction: dict[str, object] | None = None
    compactions: list[CompactionRecord] = []
    model: str | None = None
    for event in events:
        if event.kind == "user_exchange":
            if pending is not None:
                raise SessionCorruptError("User exchange appeared before pending tool results")
            conversation.append(_decode_user_exchange(event.payload))
            conversation_models.append(model)
        elif event.kind == "assistant_exchange":
            if pending is not None:
                raise SessionCorruptError("Assistant exchange appeared before pending tool results")
            assistant = _decode_assistant_exchange(event.payload)
            if assistant.tool_uses:
                pending = assistant
                pending_model = model
            else:
                conversation.append(assistant)
                conversation_models.append(model)
        elif event.kind == "tool_continuation":
            continuation = _decode_tool_continuation(event.payload)
            if pending is None or continuation.assistant != pending:
                raise SessionCorruptError("Tool continuation does not match its assistant exchange")
            conversation.append(continuation)
            conversation_models.append(pending_model)
            pending = None
            pending_model = None
        elif event.kind == "compaction_completed":
            compaction = _require_dict(event.payload, "compaction checkpoint")
            compactions.append(CompactionRecord(len(conversation), compaction))
        elif event.kind == "model_changed":
            model = _require_string(_require_mapping(event.payload, "model change"), "current")
    return conversation, conversation_models, pending, pending_model, compaction, model, compactions


def _encode_payload(payload: object) -> object:
    if isinstance(payload, UserExchange):
        return {"type": "user_exchange", "content": payload.content}
    if isinstance(payload, AssistantExchange):
        return _encode_assistant_exchange(payload)
    if isinstance(payload, ToolContinuationExchange):
        return {
            "type": "tool_continuation",
            "assistant": _encode_assistant_exchange(payload.assistant),
            "results": [
                {
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                    "is_error": result.is_error,
                    "metadata": result.metadata,
                }
                for result in payload.results
            ],
        }
    return payload


def _encode_assistant_exchange(exchange: AssistantExchange) -> dict[str, object]:
    return {
        "type": "assistant_exchange",
        "blocks": [_encode_content_block(block) for block in exchange.blocks],
        "stop_reason": exchange.stop_reason,
        "usage": exchange.usage,
    }


def _encode_content_block(block: ContentBlock) -> dict[str, object]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
            "raw": block.raw,
        }
    if isinstance(block, RedactedThinkingBlock):
        return {"type": "redacted_thinking", "data": block.data, "raw": block.raw}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "call_id": block.call_id,
            "name": block.name,
            "input": block.input,
            "raw": block.raw,
        }
    if isinstance(block, UnknownProviderBlock):
        return {"type": "unknown", "block_type": block.block_type, "raw": block.raw}
    raise TypeError(f"Unsupported content block: {type(block).__name__}")


def _decode_user_exchange(value: object) -> UserExchange:
    mapping = _require_mapping(value, "user exchange")
    return UserExchange(_require_string(mapping, "content"))


def _decode_assistant_exchange(value: object) -> AssistantExchange:
    mapping = _require_mapping(value, "assistant exchange")
    raw_blocks = mapping.get("blocks")
    if not isinstance(raw_blocks, list):
        raise SessionCorruptError("Assistant blocks must be a list")
    stop_reason = _require_string(mapping, "stop_reason")
    if stop_reason not in {"end_turn", "tool_use", "max_tokens", "refusal"}:
        raise SessionCorruptError("Assistant stop_reason is invalid")
    usage_value = mapping.get("usage", {})
    if not isinstance(usage_value, dict) or not all(
        isinstance(key, str) and isinstance(item, int) for key, item in usage_value.items()
    ):
        raise SessionCorruptError("Assistant usage is invalid")
    return AssistantExchange(
        blocks=tuple(_decode_content_block(item) for item in raw_blocks),
        stop_reason=cast(StopReason, stop_reason),
        usage=cast(dict[str, int], usage_value),
    )


def _decode_content_block(value: object) -> ContentBlock:
    mapping = _require_mapping(value, "content block")
    block_type = _require_string(mapping, "type")
    if block_type == "text":
        return TextBlock(_require_string(mapping, "text"))
    if block_type == "thinking":
        signature = mapping.get("signature")
        if signature is not None and not isinstance(signature, str):
            raise SessionCorruptError("Thinking signature is invalid")
        return ThinkingBlock(
            _require_string(mapping, "thinking"),
            signature=signature,
            raw=_require_dict(mapping.get("raw", {}), "thinking raw"),
        )
    if block_type == "redacted_thinking":
        return RedactedThinkingBlock(
            _require_string(mapping, "data"),
            raw=_require_dict(mapping.get("raw", {}), "redacted thinking raw"),
        )
    if block_type == "tool_use":
        return ToolUseBlock(
            _require_string(mapping, "call_id"),
            _require_string(mapping, "name"),
            _require_dict(mapping.get("input"), "tool input"),
            _require_dict(mapping.get("raw", {}), "tool raw"),
        )
    if block_type == "unknown":
        return UnknownProviderBlock(
            _require_string(mapping, "block_type"),
            _require_dict(mapping.get("raw"), "unknown block raw"),
        )
    raise SessionCorruptError(f"Unknown stored content block type: {block_type}")


def _decode_tool_continuation(value: object) -> ToolContinuationExchange:
    mapping = _require_mapping(value, "tool continuation")
    assistant = _decode_assistant_exchange(mapping.get("assistant"))
    raw_results = mapping.get("results")
    if not isinstance(raw_results, list):
        raise SessionCorruptError("Tool results must be a list")
    results = []
    for raw_result in raw_results:
        result = _require_mapping(raw_result, "tool result")
        is_error = result.get("is_error")
        if not isinstance(is_error, bool):
            raise SessionCorruptError("Tool result is_error must be boolean")
        results.append(
            ToolResultBlock(
                _require_string(result, "tool_use_id"),
                _require_string(result, "content"),
                is_error,
                _require_dict(result.get("metadata", {}), "tool result metadata"),
            )
        )
    try:
        return ToolContinuationExchange(assistant, tuple(results))
    except ValueError as error:
        raise SessionCorruptError("Stored tool result IDs do not match calls") from error


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SessionCorruptError(f"Stored {label} must be an object")
    return cast(dict[str, object], value)


def _require_dict(value: object, label: str) -> dict[str, object]:
    return _require_mapping(value, label)


def _require_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise SessionCorruptError(f"Stored {key} must be a string")
    return value
