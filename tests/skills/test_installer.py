from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from coding_agent.skills.installer import SkillInstaller


def _make_skill(directory: Path, name: str) -> None:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}\n---\n\nBody for {name}.\n",
        encoding="utf-8",
    )


def test_uninstall_removes_a_skill_from_project_dir(tmp_path: Path) -> None:
    project_dir = tmp_path / ".agents" / "skills"
    project_dir.mkdir(parents=True)
    _make_skill(project_dir, "demo")

    installer = SkillInstaller(
        user_dir=tmp_path / "user" / "skills",
        project_dir=project_dir,
    )

    assert asyncio.run(installer.uninstall("demo")) is True
    assert not (project_dir / "demo").exists()
    assert asyncio.run(installer.uninstall("demo")) is False


def test_reload_picks_up_newly_added_skills(tmp_path: Path) -> None:
    project_dir = tmp_path / ".agents" / "skills"
    project_dir.mkdir(parents=True)
    user_dir = tmp_path / "user" / "skills"

    installer = SkillInstaller(user_dir=user_dir, project_dir=project_dir)

    snapshot = installer.reload()
    assert {s.name for s in snapshot.skills if s.source == "project"} == set()

    _make_skill(project_dir, "demo")
    snapshot = installer.reload()
    assert {s.name for s in snapshot.skills if s.source == "project"} == {"demo"}


def test_existing_skill_names_scans_both_dirs(tmp_path: Path) -> None:
    project_dir = tmp_path / ".agents" / "skills"
    user_dir = tmp_path / "user" / "skills"
    project_dir.mkdir(parents=True)
    user_dir.mkdir(parents=True)
    _make_skill(project_dir, "proj-skill")
    _make_skill(user_dir, "user-skill")

    installer = SkillInstaller(user_dir=user_dir, project_dir=project_dir)
    assert installer._existing_skill_names() == {"proj-skill", "user-skill"}


def test_install_returns_error_without_npx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    installer = SkillInstaller(
        user_dir=tmp_path / "user" / "skills",
        project_dir=tmp_path / ".agents" / "skills",
    )

    result = asyncio.run(installer.install("owner/repo"))
    assert result.installed == ()
    assert "npx is not installed" in result.message


def test_install_streams_output_and_detects_new_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock npx to verify the output collection + detection logic."""
    project_dir = tmp_path / ".agents" / "skills"
    project_dir.mkdir(parents=True)

    class _FakeStream:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = list(lines)

        async def __aiter__(self):
            for line in self._lines:
                yield line

    class _FakeProcess:
        def __init__(self, exit_code: int, project_dir: Path) -> None:
            self.returncode: int | None = None
            self._exit_code = exit_code
            self._project_dir = project_dir
            self.stdout = _FakeStream([b"Adding skill demo\n", b"Done\n"])

        async def wait(self) -> int:
            _make_skill(self._project_dir, "demo")
            self.returncode = self._exit_code
            return self._exit_code

        def kill(self) -> None:
            pass

    async def _fake_create_subprocess_exec(*args: str, **kwargs):
        del args, kwargs
        return _FakeProcess(0, project_dir)

    monkeypatch.setattr(shutil, "which", lambda _cmd: "/fake/npx")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    installer = SkillInstaller(
        user_dir=tmp_path / "user" / "skills",
        project_dir=project_dir,
    )

    result = asyncio.run(installer.install("owner/repo"))
    assert result.installed == ("demo",)
    assert len(result.output) >= 2
    assert any("Adding skill" in line for line in result.output)
