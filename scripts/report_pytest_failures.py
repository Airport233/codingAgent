from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def _escape_workflow_command(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _emit(title: str, details: str) -> None:
    safe_title = _escape_workflow_command(title).replace(":", "%3A").replace(",", "%2C")
    print(f"::error title={safe_title}::{_escape_workflow_command(details[-4000:])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report JUnit failures as GitHub annotations.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    try:
        root = ET.parse(args.report).getroot()
    except (OSError, ET.ParseError) as error:
        details = str(error)
        if args.log is not None and args.log.is_file():
            details = args.log.read_text(encoding="utf-8", errors="replace")
        _emit("pytest failed before producing JUnit results", details)
        return

    found = False
    for case in root.iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if failure is None:
            continue
        found = True
        name = f"{case.get('classname', '')}::{case.get('name', 'unknown')}".strip(":")
        details = (failure.text or failure.get("message") or "pytest failed").strip()
        _emit(name, details)
    if not found and args.log is not None and args.log.is_file():
        _emit(
            "pytest failed without a JUnit failure entry",
            args.log.read_text(encoding="utf-8", errors="replace"),
        )


if __name__ == "__main__":
    main()
