from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.skills import SkillLoader


def write_skill(
    directory: Path,
    name: str,
    description: str,
    body: str,
    *,
    frontmatter: str = "",
) -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{frontmatter}---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_loader_discovers_standard_skill_directories_and_sorts_metadata(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    write_skill(builtin, "test-fix", "Fix a failing test", "Reproduce, fix, and verify.")
    write_skill(builtin, "code-review", "Review a change", "Report concrete findings.")

    snapshot = SkillLoader(builtin).load()

    assert [skill.name for skill in snapshot.skills] == ["code-review", "test-fix"]
    assert snapshot.skills[0].source == "builtin"
    assert snapshot.skills[1].path.name == "SKILL.md"
    assert snapshot.skills[1].root.name == "test-fix"
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
    assert "project body" in snapshot.skills[0].render_instructions()
    assert snapshot.skills[0].source == "project"


def test_loader_reports_missing_entrypoint_and_directory_name_mismatch(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    write_skill(builtin, "valid", "Valid workflow", "Do useful work.")
    (builtin / "missing-entrypoint").mkdir(parents=True)
    mismatched = write_skill(builtin, "wrong-directory", "Invalid name", "body")
    mismatched.write_text(
        "---\nname: different-name\ndescription: Invalid name\n---\nbody\n",
        encoding="utf-8",
    )

    snapshot = SkillLoader(builtin).load()

    assert [skill.name for skill in snapshot.skills] == ["valid"]
    assert len(snapshot.warnings) == 2
    assert any("missing SKILL.md" in warning for warning in snapshot.warnings)
    assert any("match the parent directory" in warning for warning in snapshot.warnings)


def test_loader_reports_legacy_single_file_skill_layout(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "old-skill.md").write_text("legacy", encoding="utf-8")

    snapshot = SkillLoader(builtin).load()

    assert snapshot.skills == ()
    assert "use <skill-name>/SKILL.md" in snapshot.warnings[0]


def test_loader_rejects_oversized_and_symlinked_skill_directories(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    oversized = write_skill(builtin, "oversized", "Too large", "x" * 500)
    outside = tmp_path / "outside" / "external"
    write_skill(tmp_path / "outside", "external", "Outside", "must not load")
    builtin.mkdir(exist_ok=True)
    linked = builtin / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        linked = None

    snapshot = SkillLoader(builtin, max_bytes=128).load()

    assert snapshot.skills == ()
    assert any(
        oversized.parent.name in warning and "too large" in warning for warning in snapshot.warnings
    )
    if linked is not None:
        assert any(
            "linked" in warning and "symbolic link" in warning for warning in snapshot.warnings
        )


def test_skill_body_is_loaded_only_when_activated(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    entrypoint = write_skill(
        builtin, "test-fix", "Fix tests", "Original instructions should not be cached."
    )
    snapshot = SkillLoader(builtin).load()
    entrypoint.write_text(
        "---\nname: test-fix\ndescription: Fix tests\n---\n\nRead lazily after activation.\n",
        encoding="utf-8",
    )

    skill = snapshot.get("test-fix")

    assert skill is not None
    rendered = skill.render_instructions()
    assert '<active_skill name="test-fix" source="builtin">' in rendered
    assert "does not grant additional tool permissions" in rendered
    assert "Read lazily after activation." in rendered
    assert "Original instructions" not in rendered
    assert snapshot.get("missing") is None


def test_invalid_body_encoding_does_not_block_metadata_discovery(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    skill_dir = builtin / "binary-body"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_bytes(
        b"---\nname: binary-body\ndescription: Discover metadata first\n---\n\xff"
    )

    snapshot = SkillLoader(builtin).load()

    assert [skill.name for skill in snapshot.skills] == ["binary-body"]
    with pytest.raises(UnicodeDecodeError):
        snapshot.skills[0].render_instructions()


def test_loader_supports_standard_optional_frontmatter_and_catalog_disclosure(
    tmp_path: Path,
) -> None:
    builtin = tmp_path / "builtin"
    write_skill(
        builtin,
        "test-fix",
        "Fix tests when CI or a local test fails",
        "Reproduce first.",
        frontmatter=(
            "license: Apache-2.0\n"
            "compatibility: Requires Python 3.12+\n"
            "metadata:\n  author: codingAgent\n  version: '1'\n"
            "allowed-tools: read_file shell\n"
        ),
    )

    snapshot = SkillLoader(builtin).load()
    skill = snapshot.skills[0]

    assert skill.license == "Apache-2.0"
    assert skill.compatibility == "Requires Python 3.12+"
    assert skill.metadata == (("author", "codingAgent"), ("version", "1"))
    assert skill.allowed_tools == "read_file shell"
    catalog = snapshot.render_catalog()
    assert "<name>test-fix</name>" in catalog
    assert "<description>Fix tests when CI or a local test fails</description>" in catalog
    assert "<location>" in catalog
    assert "Reproduce first" not in catalog


def test_default_loader_exposes_builtin_coding_workflows(tmp_path: Path) -> None:
    snapshot = SkillLoader.default(
        user_dir=tmp_path / "user", project_dir=tmp_path / "project"
    ).load()

    assert [skill.name for skill in snapshot.skills] == [
        "code-review",
        "github-delivery",
        "project-init",
        "project-map",
        "test-fix",
    ]
    assert all(skill.source == "builtin" for skill in snapshot.skills)
