from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPOSITORY_ROOT / "examples" / "acceptance" / "failing-discount"
DEFAULT_DESTINATION = REPOSITORY_ROOT / ".tmp" / "agent-acceptance"


def prepare_project(destination: Path) -> Path:
    resolved = destination.resolve()
    if resolved.exists():
        raise FileExistsError(f"Destination already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_ROOT, resolved)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an isolated project for the codingAgent acceptance test."
    )
    parser.add_argument("destination", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    try:
        destination = prepare_project(args.destination)
    except FileExistsError as error:
        parser.error(str(error))
    print(f"Prepared acceptance project: {destination}")
    print("Initial check: uv run python -m unittest discover -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
