from __future__ import annotations

import sys
from pathlib import Path

from coding_agent.prompts import render_base_prompt


def test_render_base_prompt_includes_workspace_and_platform() -> None:
    prompt = render_base_prompt(Path("/tmp/example"))

    assert "/tmp/example" in prompt
    assert sys.platform in prompt
    assert "relative to the working directory" in prompt
    assert "Never delete files" in prompt
