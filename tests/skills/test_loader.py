from __future__ import annotations

from pathlib import Path

from coding_agent.skills import SkillLoader


def write_skill(directory: Path, name: str, description: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(
        f'+++\nname = "{name}"\ndescription = "{description}"\n+++\n\n{body}\n',
        encoding="utf-8",
    )
    return path


def test_loader_discovers_valid_markdown_skills_and_sorts_by_name(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    write_skill(builtin, "test-fix", "Fix a failing test", "Reproduce, fix, and verify.")
    write_skill(builtin, "code-review", "Review a change", "Report concrete findings.")

    snapshot = SkillLoader(builtin).load()

    assert [skill.name for skill in snapshot.skills] == ["code-review", "test-fix"]
    assert snapshot.skills[0].source == "builtin"
    assert snapshot.skills[1].instructions == "Reproduce, fix, and verify."
    assert snapshot.warnings == ()


def test_project_and_user_skills_override_lower_priority_definitions(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    project = tmp_path / "project"
    write_skill(builtin, "test-fix", "Builtin", "builtin body")
    write_skill(user, "test-fix", "User", "user body")
    write_skill(project, "test-fix", "Project", "project body")

    snapshot = SkillLoader(builtin, user_dir=user, project_dir=project).load()

    assert len(snapshot.skills) == 1
    assert snapshot.skills[0].description == "Project"
    assert snapshot.skills[0].instructions == "project body"
    assert snapshot.skills[0].source == "project"


def test_loader_reports_bad_metadata_without_hiding_other_skills(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    write_skill(builtin, "valid", "Valid workflow", "Do useful work.")
    (builtin / "broken.md").write_text("# no metadata\n", encoding="utf-8")
    write_skill(builtin, "Bad Name", "Invalid name", "body")

    snapshot = SkillLoader(builtin).load()

    assert [skill.name for skill in snapshot.skills] == ["valid"]
    assert len(snapshot.warnings) == 2
    assert all("broken.md" in warning or "Bad Name.md" in warning for warning in snapshot.warnings)


def test_loader_rejects_oversized_and_symlinked_skills(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    oversized = write_skill(builtin, "oversized", "Too large", "x" * 500)
    outside = write_skill(tmp_path / "outside", "external", "Outside", "must not load")
    linked = builtin / "linked.md"
    try:
        linked.symlink_to(outside)
    except OSError:
        linked = None

    snapshot = SkillLoader(builtin, max_bytes=128).load()

    assert snapshot.skills == ()
    assert any(
        oversized.name in warning and "too large" in warning for warning in snapshot.warnings
    )
    if linked is not None:
        assert any(
            "linked.md" in warning and "symbolic link" in warning for warning in snapshot.warnings
        )


def test_snapshot_supports_lookup_and_prompt_rendering(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    write_skill(builtin, "test-fix", "Fix tests", "Always reproduce the failure first.")
    snapshot = SkillLoader(builtin).load()

    skill = snapshot.get("test-fix")

    assert skill is not None
    rendered = skill.render_instructions()
    assert '<active_skill name="test-fix" source="builtin">' in rendered
    assert "does not grant additional tool permissions" in rendered
    assert "Always reproduce the failure first." in rendered
    assert snapshot.get("missing") is None


def test_default_loader_exposes_builtin_coding_workflows(tmp_path: Path) -> None:
    snapshot = SkillLoader.default(
        user_dir=tmp_path / "user", project_dir=tmp_path / "project"
    ).load()

    assert [skill.name for skill in snapshot.skills] == [
        "code-review",
        "project-init",
        "project-map",
        "test-fix",
    ]
    assert all(skill.source == "builtin" for skill in snapshot.skills)
