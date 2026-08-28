# P0 verification map

This file maps the MVP requirements in `docs/requirements.md` to executable evidence. The
repository-wide quality command is the GitHub Actions `CI` workflow on macOS and Windows.

| Requirement | Evidence |
| --- | --- |
| FR-CLI-001–003 | `tests/test_project_scaffold.py`, `test_repl_accepts_multiple_turns_and_compacts_context` |
| FR-CLI-004–005A | `test_repl_shows_shell_details_and_toggles_thinking`, `test_repl_hides_thinking_content_by_default` |
| FR-CLI-006 | `test_runtime_settings_use_environment_without_exposing_private_values` |
| FR-CLI-007 | `test_cancelling_a_running_tool_is_recorded_before_propagation`, `test_cancelling_a_provider_request_keeps_the_session_usable` |
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

On 2026-08-28, an initial live run with the configured Anthropic-compatible Provider completed the
failing-discount task, but it also exposed that a nested `--workspace` was incorrectly widened to
its parent Git repository. That run does not count as an isolated acceptance result. The workspace
boundary and session identity were corrected and covered by a cross-platform regression test. A
post-fix live rerun was blocked before inference by the gateway reporting that the configured model
identifier was invalid, so the isolated live acceptance below remains pending. The tracked failing
fixture was restored and still fails its percentage-discount test as intended.

Use an isolated copy for subsequent runs:

```bash
uv run python scripts/prepare_acceptance_project.py .tmp/agent-acceptance-manual
uv run coding-agent --model claude-sonnet-5 --workspace .tmp/agent-acceptance-manual
```

Submit the task from `.tmp/agent-acceptance-manual/TASK.md`, then exercise `/thinking`, `/context`,
`/compact`, `/model`, `/clear`, and `/exit`. Never place Provider credentials or private endpoints
in the command transcript.
