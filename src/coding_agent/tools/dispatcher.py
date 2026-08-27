from __future__ import annotations

from pydantic import ValidationError

from coding_agent.domain import ToolResultBlock, ToolUseBlock
from coding_agent.tools.base import RecoverableToolError
from coding_agent.tools.catalog import ToolCatalog


class ToolDispatcher:
    def __init__(self, catalog: ToolCatalog) -> None:
        self.catalog = catalog

    async def execute(self, call: ToolUseBlock) -> ToolResultBlock:
        tool = self.catalog.get(call.name)
        if tool is None:
            return ToolResultBlock(
                tool_use_id=call.call_id,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )
        try:
            arguments = tool.input_model.model_validate(call.input)
            output = await tool.execute(arguments)
        except (ValidationError, RecoverableToolError) as error:
            return ToolResultBlock(
                tool_use_id=call.call_id,
                content=str(error),
                is_error=True,
            )
        return ToolResultBlock(
            tool_use_id=call.call_id,
            content=output.content,
            is_error=False,
            metadata=output.metadata,
        )
