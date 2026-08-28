from __future__ import annotations

import argparse
import re
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        r"\s*[:=]\s*['\"]?(?!(?:\$?\{|example|dummy|fake))[^\s'\"]{12,}"
    ),
)
SKIPPED_PARTS = {".git", ".venv", ".tmp", "__pycache__"}


def iter_files(paths: list[Path]):
    for path in paths:
        if path.is_file():
            yield path
            continue
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and not SKIPPED_PARTS.intersection(candidate.parts):
                    yield candidate


def contains_secret_text(content: str) -> bool:
    return any(pattern.search(content) for pattern in SECRET_PATTERNS)


def contains_secret(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return contains_secret_text(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject obvious committed credentials.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    unsafe_paths = [path for path in iter_files(args.paths) if contains_secret(path)]
    for path in unsafe_paths:
        print(f"possible secret: {path}")
    return int(bool(unsafe_paths))


if __name__ == "__main__":
    raise SystemExit(main())
