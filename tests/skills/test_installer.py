from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from coding_agent.skills.installer import SkillInstaller


def _make_skill_repo(repo_dir: Path, name: str) -> None:
    skill_dir = repo_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}\n---\n\nBody for {name}.\n",
        encoding="utf-8",
    )


def test_uninstall_removes_a_skill_from_project_dir(tmp_path: Path) -> None:
    project_dir = tmp_path / ".agents" / "skills"
    project_dir.mkdir(parents=True)
    _make_skill_repo(project_dir, "demo")

    installer = SkillInstaller(
        user_dir=tmp_path / "user" / "skills",
        project_dir=project_dir,
    )

    removed = pytest.run_sync(installer.uninstall("demo")) if hasattr(pytest, "run_sync") else True
    # installer.uninstall is async; use asyncio directly
    import asyncio

    removed = asyncio.run(installer.uninstall("demo"))
    assert removed is True
    assert not (project_dir / "demo").exists()

    # Second uninstall returns False
    assert asyncio.run(installer.uninstall("demo")) is False


def test_reload_picks_up_newly_added_skills(tmp_path: Path) -> None:
    project_dir = tmp_path / ".agents" / "skills"
    project_dir.mkdir(parents=True)
    user_dir = tmp_path / "user" / "skills"

    installer = SkillInstaller(user_dir=user_dir, project_dir=project_dir)

    snapshot = installer.reload()
    project_names = {s.name for s in snapshot.skills if s.source == "project"}
    assert project_names == set()

    _make_skill_repo(project_dir, "demo")
    snapshot = installer.reload()
    project_names = {s.name for s in snapshot.skills if s.source == "project"}
    assert project_names == {"demo"}


def test_existing_skill_names_scans_both_dirs(tmp_path: Path) -> None:
    project_dir = tmp_path / ".agents" / "skills"
    user_dir = tmp_path / "user" / "skills"
    project_dir.mkdir(parents=True)
    user_dir.mkdir(parents=True)
    _make_skill_repo(project_dir, "proj-skill")
    _make_skill_repo(user_dir, "user-skill")

    installer = SkillInstaller(user_dir=user_dir, project_dir=project_dir)
    names = installer._existing_skill_names()
    assert names == {"proj-skill", "user-skill"}


def test_install_returns_error_without_npx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    installer = SkillInstaller(
        user_dir=tmp_path / "user" / "skills",
        project_dir=tmp_path / ".agents" / "skills",
    )

    import asyncio

    result = asyncio.run(installer.install("owner/repo"))
    assert result.installed == ()
    assert "npx is not installed" in result.message
