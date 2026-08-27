from __future__ import annotations

from collections.abc import Sequence

from coding_agent.tools.base import Tool, ToolSource, ToolSpec


class ToolCatalog:
    def __init__(self, tools: dict[str, Tool]) -> None:
        self._tools = tools

    @classmethod
    async def create(cls, sources: Sequence[ToolSource]) -> ToolCatalog:
        tools: dict[str, Tool] = {}
        for source in sources:
            for tool in await source.list_tools():
                if tool.name in tools:
                    raise ValueError(f"duplicate tool name: {tool.name}")
                tools[tool.name] = tool
        return cls(tools)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            ToolSpec(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_model.model_json_schema(),
            )
            for tool in self._tools.values()
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)
