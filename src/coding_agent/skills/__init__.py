from coding_agent.skills.loader import SkillDefinition, SkillLoader, SkillSnapshot


def format_skill_list(
    skills: tuple[tuple[str, str, str], ...], warnings: tuple[str, ...] = ()
) -> str:
    if not skills:
        lines = ["No coding workflows are available."]
    else:
        width = max(len(name) for name, _description, _source in skills)
        lines = ["Skills:"]
        lines.extend(
            f"  {name:<{width}}  [{source}] {description}" for name, description, source in skills
        )
        lines.append("Use /skill <name> <task> to activate one workflow for a task.")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    return "\n".join(lines)


__all__ = ["SkillDefinition", "SkillLoader", "SkillSnapshot", "format_skill_list"]
