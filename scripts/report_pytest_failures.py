from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def _escape_workflow_command(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report JUnit failures as GitHub annotations.")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    root = ET.parse(args.report).getroot()
    for case in root.iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if failure is None:
            continue
        name = f"{case.get('classname', '')}::{case.get('name', 'unknown')}".strip(":")
        details = (failure.text or failure.get("message") or "pytest failed").strip()
        print(
            f"::error title={_escape_workflow_command(name)}::"
            f"{_escape_workflow_command(details[-4000:])}"
        )


if __name__ == "__main__":
    main()
