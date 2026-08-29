# P0 verification map

This file maps the MVP requirements in `docs/requirements.md` to executable evidence. The
repository-wide quality command is the GitHub Actions `CI` workflow on macOS and Windows.

| Requirement | Evidence |
| --- | --- |
| FR-CLI-001–003 | `tests/test_project_scaffold.py`, `tests/test_tui.py` |
| FR-CLI-004–005A | `test_tui_renders_collapsible_thinking_and_completed_tool_card`, `test_tui_slash_commands_update_state_without_leaving_full_screen` |
| FR-CLI-006 | `test_runtime_settings_use_environment_without_exposing_private_values` |
| FR-CLI-007 | `test_ctrl_c_cancels_active_provider_without_closing_tui`, `test_cancelling_a_running_tool_is_recorded_before_propagation` |
| FR-CLI-008 | composer wrapping, multiline submission and `test_empty_composer_navigates_prompt_history_and_text_arrows_move_to_edges` |
| FR-CLI-009–010 | conversation/input selection-copy tests, CLI `--prompt` integration tests |
| FR-CLI-011 | `test_slash_popup_filters_navigates_completes_and_dismisses`, `test_model_command_opens_secondary_picker_and_switches_selection` |
| FR-SESS-006, FR-CMD-008 | `test_sessions_can_be_listed_and_resumed_by_id`, `test_resume_command_opens_full_screen_picker_and_installs_selection`, `test_resume_is_explicitly_blocked_while_a_turn_is_running` |
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

## Manual findings

Recorded on 2026-08-28 for follow-up on `feature/cli-hardening`:

- In the tested macOS terminal, copy and paste currently work through `Ctrl+C`/`Ctrl+V`, but the
  conventional `Command+C`/`Command+V` behavior is not yet verified as working end to end.
- Resolved on `feature/cli-hardening`: the TUI context label now refreshes after the final
  assistant exchange and manual `/compact`, and labels its current-context value as an estimate.
- Resolved on `feature/cli-hardening`: projected history now wraps the summary in an explicit
  codingAgent-generated checkpoint marker. Request-shape tests, rather than model self-reporting,
  remain the acceptance evidence for compaction.

Recorded on 2026-08-29, unresolved:

- Provider stream failures are not diagnosable. When an endpoint closes the SSE stream without
  sending `message_stop`, `AnthropicStreamAggregator` never emits `ProviderResponseFinished`, no
  exception is raised, and `application.py:269` reports only
  `Provider response ended without a completed exchange`. The message carries no HTTP status, no
  response body and no endpoint identity, so it is indistinguishable from a wrong model name, an
  out-of-quota endpoint and a genuinely protocol-incompatible Provider. This fails the error-message
  rule in `docs/requirements.md` ("错误信息应包含操作类型、可恢复性和建议动作"): it states neither
  recoverability nor a suggested action. `providers/anthropic.py:65` is the only layer that still
  holds the HTTP response context and is therefore where the detail must be captured. Any fix must
  satisfy NFR-SEC-003 and redact credentials, auth headers and private endpoints, which is precisely
  why the current message was written to be contentless.
- Troubleshooting note for the above: check the environment before suspecting the Provider.
  `CODING_AGENT_BASE_URL` and `CODING_AGENT_API_KEY` take precedence over the user-level
  `config.toml` at `runtime.py:137` and `runtime.py:143`, so a configured provider profile can be
  silently overridden and the request can reach an entirely different endpoint than the one the
  session log's `model_changed` event implies. The session log deliberately does not record the base
  URL, so this override is invisible in the transcript.
