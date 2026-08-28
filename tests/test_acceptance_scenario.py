from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.prepare_acceptance_project import prepare_project


def test_prepared_acceptance_project_starts_with_one_failing_behavior(tmp_path: Path) -> None:
    destination = tmp_path / "discount-service"

    prepare_project(destination)
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v"],
        cwd=destination,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert "test_percentage_discount" in result.stderr
    assert "FAILED (failures=1)" in result.stderr
    assert (destination / "CODING_AGENT.md").is_file()
    assert (destination / "TASK.md").is_file()


def test_prepare_project_refuses_to_replace_existing_directory(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_project(destination)
