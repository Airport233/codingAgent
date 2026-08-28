# P0 verification map

This file maps the MVP requirements in `docs/requirements.md` to executable evidence. The
repository-wide quality command is the GitHub Actions `CI` workflow on macOS and Windows.

| Requirement | Evidence |
| --- | --- |
| FR-CLI-001–003 | `tests/test_project_scaffold.py`, `tests/test_tui.py` |
| FR-CLI-004–005A | `test_tui_renders_collapsible_thinking_and_completed_tool_card`, `test_tui_slash_commands_update_state_without_leaving_full_screen` |
| FR-CLI-006 | `test_runtime_settings_use_environment_without_exposing_private_values` |
| FR-CLI-007 | `test_ctrl_c_cancels_active_provider_without_closing_tui`, `test_cancelling_a_running_tool_is_recorded_before_propagation` |
| FR-CLI-008 | `test_composer_soft_wraps_long_input_instead_of_scrolling_horizontally`, `test_shift_enter_adds_newline_and_enter_submits_multiline_prompt` |
| FR-CLI-009–010 | `test_ctrl_c_copies_selected_conversation_text_when_idle`, CLI `--prompt` integration tests |
| FR-PROV-001–006 | `test_runtime_settings_load_user_provider_profile`, `test_repl_lists_and_switches_models_then_starts_a_new_session` |
| FR-PROV-007, FR-PROV-010–012 | `tests/providers/`, `test_runtime_resume_with_new_model_excludes_old_thinking_from_requests` |
| FR-AGENT-001–007 | `tests/test_walking_skeleton.py`, including multi-tool, cancellation and recoverable-error cases |
| FR-TOOL-READ-001–003 | `tests/tools/test_read_search.py` |
| FR-TOOL-WRITE-001–004, FR-TOOL-DIR-001 | `tests/tools/test_write_edit.py` |
| FR-TOOL-EDIT-001–003 | `tests/tools/test_write_edit.py` |
| FR-TOOL-SEARCH-001–004 | `tests/tools/test_read_search.py` |
| FR-TOOL-SHELL-001–006 | `tests/tools/test_shell.py`, plus Windows/macOS CI jobs |
| FR-CTX-001–009 | `tests/context/test_context_manager.py`, `test_repl_accepts_multiple_turns_and_compacts_context`, `test_runtime_resume_restores_the_latest_compaction_checkpoint` |
| FR-MEM-001–005 | `tests/memory/test_memory_loader.py` |
| FR-SESS-001–005, FR-SESS-007–010 | `tests/sessions/test_jsonl_sessions.py`, runtime resume and redaction tests |
| FR-CMD-001–007 | `tests/test_runtime_integration.py` |
| FR-EXT-001–002, FR-EXT-005 | application event tests, Provider adapter tests and `ToolSource` catalog tests |

## Manual acceptance

On 2026-08-28, the full-screen TUI completed the failing-discount task in an isolated workspace
using the configured Anthropic-compatible Provider and `claude-sonnet-4-6`. The agent read the task,
fixed only the production calculation, and the three fixture tests passed in an independent rerun.
The run also verified full-screen rendering, collapsible tool cards, a persistent composer, and clean
exit with Ctrl+Q. The tracked failing fixture remains unchanged and still fails as intended.

Use an isolated copy for subsequent runs:

```bash
uv run python scripts/prepare_acceptance_project.py .tmp/agent-acceptance-manual
uv run coding-agent --model claude-sonnet-4-6 --workspace .tmp/agent-acceptance-manual
```

Submit the task from `.tmp/agent-acceptance-manual/TASK.md`, then exercise `/thinking`, `/context`,
`/compact`, `/model`, `/clear`, and `/exit`. Never place Provider credentials or private endpoints
in the command transcript.
