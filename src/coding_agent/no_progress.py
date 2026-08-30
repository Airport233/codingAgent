from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from coding_agent.domain import ToolResultBlock, ToolUseBlock

NO_PROGRESS_WARNING = (
    "\n\n[NO_PROGRESS_WARNING]\n"
    "This exact tool call produced the same result twice consecutively. "
    "Do not repeat it unchanged; gather new evidence or change approach."
)


@dataclass(frozen=True, slots=True)
class NoProgressObservation:
    action: Literal["none", "warn", "stop"]
    repetition_count: int
    fingerprint: str


class NoProgressDetector:
    """Detect consecutive tool calls that produce no observable progress."""

    def __init__(self, *, warn_at: int = 2, stop_at: int = 3) -> None:
        if warn_at < 2 or stop_at <= warn_at:
            raise ValueError("no-progress thresholds must satisfy 2 <= warn_at < stop_at")
        self._warn_at = warn_at
        self._stop_at = stop_at
        self._last_fingerprint: str | None = None
        self._repetition_count = 0

    def observe(self, call: ToolUseBlock, result: ToolResultBlock) -> NoProgressObservation:
        fingerprint = _fingerprint(call, result)
        if fingerprint == self._last_fingerprint:
            self._repetition_count += 1
        else:
            self._last_fingerprint = fingerprint
            self._repetition_count = 1

        if self._repetition_count >= self._stop_at:
            action: Literal["none", "warn", "stop"] = "stop"
        elif self._repetition_count == self._warn_at:
            action = "warn"
        else:
            action = "none"
        return NoProgressObservation(action, self._repetition_count, fingerprint)


def _fingerprint(call: ToolUseBlock, result: ToolResultBlock) -> str:
    payload = {
        "tool": call.name,
        "arguments": call.input,
        "content": _stable_result_content(call.name, result.content),
        "is_error": result.is_error,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _stable_result_content(tool_name: str, content: str) -> str:
    if tool_name != "shell":
        return content
    return re.sub(r"(?m)^duration_ms:\s*\d+\s*$", "duration_ms: <ignored>", content)
