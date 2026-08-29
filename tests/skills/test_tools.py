from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.domain import ToolUseBlock
from coding_agent.skills import SkillLoader
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tools.skills import SkillToolSource


def write_skill(directory: Path, name: str, description: str, body: str) -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_skill_tools_activate_metadata_then_read_a_reference_on_demand(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    entrypoint = write_skill(
        skills_root,
        "project-init",
        "Draft project guidance when initialization is requested",
        "Read references/template.md only when drafting output.",
    )
    references = entrypoint.parent / "references"
    references.mkdir()
    (references / "template.md").write_text("# Draft template\n", encoding="utf-8")
    assets = entrypoint.parent / "assets"
    assets.mkdir()
    (assets / "example.txt").write_text("example asset\n", encoding="utf-8")
    snapshot = SkillLoader(skills_root).load()
    catalog = await ToolCatalog.create((SkillToolSource(snapshot),))
    dispatcher = ToolDispatcher(catalog)

    activated = await dispatcher.execute(
        ToolUseBlock("activate", "activate_skill", {"name": "project-init"})
    )
    resource = await dispatcher.execute(
        ToolUseBlock(
            "resource",
            "read_skill_resource",
            {"skill_name": "project-init", "path": "references/template.md"},
        )
    )
    asset = await dispatcher.execute(
        ToolUseBlock(
            "asset",
            "read_skill_resource",
            {"skill_name": "project-init", "path": "assets/example.txt"},
        )
    )

    assert activated.is_error is False
    assert activated.metadata == {"skill_name": "project-init", "source": "builtin"}
    assert resource.content == "# Draft template\n"
    assert asset.content == "example asset\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["../outside.md", "SKILL.md", "other/private.txt", "references/missing.md"],
)
async def test_skill_resource_tool_rejects_unsafe_or_unsupported_paths(
    tmp_path: Path, path: str
) -> None:
    skills_root = tmp_path / "skills"
    entrypoint = write_skill(skills_root, "safe-skill", "A safe skill", "Use references.")
    assets = entrypoint.parent / "assets"
    assets.mkdir()
    (assets / "private.txt").write_text("not readable", encoding="utf-8")
    snapshot = SkillLoader(skills_root).load()
    dispatcher = ToolDispatcher(await ToolCatalog.create((SkillToolSource(snapshot),)))

    result = await dispatcher.execute(
        ToolUseBlock(
            "resource",
            "read_skill_resource",
            {"skill_name": "safe-skill", "path": path},
        )
    )

    assert result.is_error is True


@pytest.mark.asyncio
async def test_skill_resource_tool_enforces_size_limit(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    entrypoint = write_skill(skills_root, "safe-skill", "A safe skill", "Use references.")
    references = entrypoint.parent / "references"
    references.mkdir()
    (references / "large.md").write_text("x" * 32, encoding="utf-8")
    snapshot = SkillLoader(skills_root, max_resource_bytes=16).load()
    dispatcher = ToolDispatcher(await ToolCatalog.create((SkillToolSource(snapshot),)))

    result = await dispatcher.execute(
        ToolUseBlock(
            "resource",
            "read_skill_resource",
            {"skill_name": "safe-skill", "path": "references/large.md"},
        )
    )

    assert result.is_error is True
    assert "too large" in result.content
