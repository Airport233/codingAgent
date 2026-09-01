from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable, Sequence

from coding_agent.approval import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalPolicy,
    ConfigurableApprovalPolicy,
)
from coding_agent.context import CompactionCheckpoint, ContextManager, ContextStatus
from coding_agent.domain import (
    AssistantExchange,
    CompactionRecord,
    Conversation,
    ConversationExchange,
    ProviderContinuationExchange,
    RedactedThinkingBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
    UserExchange,
)
from coding_agent.events import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    AgentStarted,
    ApprovalRequested,
    ContextUsageChanged,
    CoreEvent,
    TextDelta,
    ThinkingDelta,
    ThinkingFinished,
    ThinkingStarted,
    ToolFinished,
    ToolStarted,
    WarningRaised,
)
from coding_agent.memory.loader import ProjectMemoryLoader
from coding_agent.no_progress import NO_PROGRESS_WARNING, NoProgressDetector
from coding_agent.providers.base import (
    Provider,
    ProviderResponseFinished,
    ProviderTextDelta,
    ProviderThinkingDelta,
    RetryableProviderError,
)
from coding_agent.sessions.base import SessionStore
from coding_agent.skills import SkillDefinition, SkillSnapshot
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tools.shell import ShellRiskVerdict


class AgentApplication:
    def __init__(
        self,
        provider: Provider,
        dispatcher: ToolDispatcher,
        sessions: SessionStore,
        max_steps: int = 20,
        memory_loader: ProjectMemoryLoader | None = None,
        initial_exchanges: Sequence[ConversationExchange] = (),
        context_manager: ContextManager | None = None,
        context_reprojected: bool = False,
        display_redactor: Callable[[object], object] | None = None,
        initial_compactions: Sequence[CompactionRecord] = (),
        approval_policy: ApprovalPolicy | None = None,
        shell_classifier: Callable[[ToolUseBlock], ShellRiskVerdict | None] | None = None,
        guardian_enabled: bool = False,
        skills: SkillSnapshot | None = None,
        base_prompt: str | None = None,
        response_retry_limit: int = 2,
    ) -> None:
        if response_retry_limit < 0:
            raise ValueError("response_retry_limit must be non-negative")
        self._base_prompt = base_prompt
        self._provider = provider
        self._dispatcher = dispatcher
        self._sessions = sessions
        self._max_steps = max_steps
        self._memory_loader = memory_loader
        self._last_memory_digest: str | None = None
        self._conversation = Conversation(list(initial_exchanges))
        self._context_manager = context_manager
        self._context_reprojected = context_reprojected
        self._display_redactor = display_redactor or (lambda value: value)
        self._compaction_history = tuple(initial_compactions)
        self._approval_policy = approval_policy or ConfigurableApprovalPolicy()
        self._shell_classifier = shell_classifier
        self._guardian_enabled = guardian_enabled
        self._skills = skills or SkillSnapshot(())
        self._response_retry_limit = response_retry_limit
        self._pending_approvals: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self._closed = False

    async def close_session(self) -> None:
        if self._closed:
            return
        await self._sessions.append("session_closed", {})
        self._closed = True

    def context_status(self) -> ContextStatus | None:
        if self._context_manager is None:
            return None
        system_instructions = self._load_system_instructions()
        return self._context_manager.status(
            self._conversation.snapshot(),
            self._supplemental_characters(system_instructions),
        )

    def context_checkpoint(self) -> CompactionCheckpoint | None:
        return None if self._context_manager is None else self._context_manager.checkpoint

    def conversation_history(self) -> tuple[ConversationExchange, ...]:
        """Return the durable, unprojected transcript for read-only presentation."""
        return self._conversation.snapshot()

    def compaction_history(self) -> tuple[CompactionRecord, ...]:
        """Return durable compaction events with their original transcript positions."""
        return self._compaction_history

    def available_skills(self) -> tuple[tuple[str, str, str], ...]:
        return tuple((skill.name, skill.description, skill.source) for skill in self._skills.skills)

    def skill_warnings(self) -> tuple[str, ...]:
        return self._skills.warnings

    def reload_skills(self, skills: SkillSnapshot) -> None:
        """Replace the live skill snapshot (after install/uninstall)."""
        self._skills = skills

    def approval_mode(self) -> ApprovalMode | None:
        """Return the live approval mode, or None if the policy doesn't expose one."""
        if isinstance(self._approval_policy, ConfigurableApprovalPolicy):
            return self._approval_policy.mode
        return None

    def set_approval_mode(self, mode: ApprovalMode) -> bool:
        """Change the approval mode in place; takes effect on the next evaluate() call.

        Returns False if the configured policy doesn't support live mode changes.
        """
        if isinstance(self._approval_policy, ConfigurableApprovalPolicy):
            self._approval_policy.mode = mode
            return True
        return False

    async def resolve_approval(self, request_id: str, decision: ApprovalDecision) -> bool:
        pending = self._pending_approvals.get(request_id)
        if pending is None or pending.done():
            return False
        await self._sessions.append(
            "approval_resolved", {"request_id": request_id, "decision": decision}
        )
        pending.set_result(decision)
        return True

    async def compact_context(
        self, reason: str = "manual", active_skill: SkillDefinition | None = None
    ) -> CompactionCheckpoint | None:
        if self._context_manager is None:
            return None
        history = self._conversation.snapshot()
        supplemental_characters = self._supplemental_characters(
            self._load_system_instructions(active_skill)
        )
        summarize = (
            None
            if self._context_manager.status(history, supplemental_characters).level == "hard"
            else self._request_context_summary
        )
        return await self._context_manager.compact(
            history,
            reason=reason,
            persist=self._sessions.append,
            summarize=summarize,
            supplemental_characters=supplemental_characters,
        )

    def _load_system_instructions(self, active_skill: SkillDefinition | None = None) -> str | None:
        sections: list[str] = []
        if self._base_prompt:
            sections.append(self._base_prompt)
        skill_catalog = self._skills.render_catalog()
        if skill_catalog:
            sections.append(skill_catalog)
        if self._memory_loader is not None:
            memory = self._memory_loader.load().rendered
            if memory:
                sections.append(memory)
        if active_skill is not None:
            sections.append(active_skill.render_instructions())
        return "\n\n".join(sections) or None

    def _supplemental_characters(self, system_instructions: str | None) -> int:
        tools = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in self._dispatcher.catalog.specs
        ]
        serialized_tools = json.dumps(tools, ensure_ascii=False, sort_keys=True)
        return len(system_instructions or "") + len(serialized_tools)

    async def _request_context_summary(self, exchanges: tuple[ConversationExchange, ...]) -> str:
        instruction = UserExchange(
            "Summarize the preceding conversation as plain text using exactly the schema "
            "below. Put each field on one line and do not add Markdown headings:\n"
            "context_summary_version: 2\n"
            "task_goal: <current objective>\n"
            "user_constraints: <active user and project constraints>\n"
            "decisions_and_rationale: <meaningful decisions>\n"
            "files_read: <relevant paths>\n"
            "files_modified: <changed paths>\n"
            "commands_and_results: <important commands and outcomes>\n"
            "verification_status: <what was actually verified>\n"
            "known_failures: <current failures or none>\n"
            "pending_work: <concrete next work>\n"
            "For every meaningful item in decisions_and_rationale, include Decision, Why, "
            "and Authority in this form: Decision: <choice>; Why: <reason>; Authority: "
            "<user|project specification|architecture|agent|unknown>. Separate multiple "
            "decisions with ' | '. Preserve user-authorized decisions over agent choices. "
            "If the reason was not stated or cannot be supported by the conversation, write "
            "'Why: not recorded'; do not invent a rationale. Keep the current decision when "
            "a newer instruction supersedes an older one. Preserve concrete paths, commands, "
            "results, constraints, failures, and unfinished work. Do not call tools."
        )
        response: AssistantExchange | None = None
        async for event in self._provider.stream((*exchanges, instruction), (), None):
            if isinstance(event, ProviderResponseFinished):
                response = event.exchange
        if response is None or response.stop_reason != "end_turn":
            raise RuntimeError("Provider did not complete the context summary")
        return response.text

    async def _request_shell_guardian_review(self, command: str) -> str | None:
        """Ask the model for a second opinion on a shell command's risk.

        This is advisory only: the note is surfaced to the human approver via
        ApprovalRequested.guardian_note and never converted into an automatic
        allow. A failure here must never block the normal approval flow.
        """
        instruction = UserExchange(
            "You are a safety reviewer, not the agent doing the task. In one or two short "
            "sentences, state what the following shell command does and any concrete risk "
            "(data loss, leaving the project directory, credential exposure). Do not approve "
            "or deny it; a human will decide. Do not call tools.\n\n"
            f"Command: {command}"
        )
        response: AssistantExchange | None = None
        async for event in self._provider.stream((instruction,), (), None):
            if isinstance(event, ProviderResponseFinished):
                response = event.exchange
        if response is None or response.stop_reason != "end_turn" or not response.text:
            return None
        return response.text

    async def run(self, prompt: str, *, skill_name: str | None = None) -> AsyncIterator[CoreEvent]:
        active_skill = None if skill_name is None else self._skills.get(skill_name)
        if skill_name is not None and active_skill is None:
            yield AgentFailed(message=f"Unknown skill: {skill_name}")
            return
        if active_skill is not None:
            await self._sessions.append(
                "skill_invoked",
                {"name": active_skill.name, "source": active_skill.source, "task": prompt},
            )
        user_exchange = UserExchange(content=prompt)
        self._conversation.exchanges.append(user_exchange)
        await self._sessions.append("user_exchange", user_exchange)
        yield AgentStarted(
            prompt=prompt,
            skill_name=active_skill.name if active_skill is not None else None,
        )
        if self._context_reprojected:
            yield WarningRaised(
                "Context was reprojected for the selected model; prior-model thinking was omitted"
            )
            self._context_reprojected = False

        no_progress = NoProgressDetector()

        for _step in range(self._max_steps):
            response: AssistantExchange | None = None
            try:
                system_instructions = self._load_system_instructions(active_skill)
            except (OSError, UnicodeError, ValueError) as error:
                message = f"Unable to load active skill: {self._redacted_text(error)}"
                await self._sessions.append(
                    "turn_failed", {"phase": "skill_activation", "message": message}
                )
                yield AgentFailed(message=message)
                return
            if self._memory_loader is not None:
                memory = self._memory_loader.load()
                if memory.digest != self._last_memory_digest:
                    await self._sessions.append(
                        "memory_snapshot_changed",
                        {
                            "digest": memory.digest,
                            "entries": [
                                {
                                    "source": entry.source,
                                    "priority": entry.priority,
                                    "content": entry.content,
                                }
                                for entry in memory.entries
                            ],
                        },
                    )
                    self._last_memory_digest = memory.digest
            if self._context_manager is not None:
                supplemental_characters = self._supplemental_characters(system_instructions)
                status = self._context_manager.status(
                    self._conversation.snapshot(), supplemental_characters
                )
                if status.level in {"soft", "hard"}:
                    try:
                        await self.compact_context(reason="auto", active_skill=active_skill)
                    except Exception:
                        yield WarningRaised(
                            "Context compaction failed; continuing with the original context"
                        )
                request_history = self._context_manager.project(self._conversation.snapshot())
                status = self._context_manager.status(
                    self._conversation.snapshot(), supplemental_characters
                )
                yield ContextUsageChanged(
                    used_tokens=status.used_tokens,
                    context_window=status.context_window,
                    level=status.level,
                )
                if status.level == "hard":
                    message = "Context remains above the safe request limit"
                    await self._sessions.append("turn_failed", {"message": message})
                    yield AgentFailed(message=message)
                    return
            else:
                request_history = self._conversation.snapshot()
            retry_guidance: str | None = None
            for request_attempt in range(self._response_retry_limit + 1):
                response = None
                thinking_active = False
                thinking_seen_live = False
                request_system_instructions = self._with_retry_guidance(
                    system_instructions, retry_guidance
                )
                try:
                    async for event in self._provider.stream(
                        request_history,
                        self._dispatcher.catalog.specs,
                        request_system_instructions,
                    ):
                        if isinstance(event, ProviderThinkingDelta):
                            thinking_seen_live = True
                            if not thinking_active:
                                thinking_active = True
                                yield ThinkingStarted()
                            yield ThinkingDelta(text=event.thinking)
                        elif isinstance(event, ProviderTextDelta):
                            if thinking_active:
                                thinking_active = False
                                yield ThinkingFinished()
                            yield TextDelta(text=event.text)
                        elif isinstance(event, ProviderResponseFinished):
                            response = event.exchange
                except asyncio.CancelledError:
                    if thinking_active:
                        yield ThinkingFinished()
                    await self._sessions.append("turn_cancelled", {"phase": "provider_request"})
                    yield AgentCancelled(message="Provider request cancelled")
                    return
                except RetryableProviderError as error:
                    if thinking_active:
                        yield ThinkingFinished()
                    if request_attempt < self._response_retry_limit:
                        retry_attempt = request_attempt + 1
                        await self._record_provider_retry(reason=error.code, attempt=retry_attempt)
                        yield WarningRaised(
                            "Provider returned an invalid response; retrying "
                            f"({retry_attempt}/{self._response_retry_limit})"
                        )
                        retry_guidance = self._protocol_retry_guidance(self._redacted_text(error))
                        continue
                    message = self._redacted_text(error)
                    failure = self._retry_exhausted_message(message)
                    await self._sessions.append(
                        "turn_failed", {"phase": "provider_request", "message": failure}
                    )
                    yield AgentFailed(message=f"Provider request failed: {failure}")
                    return
                except Exception as error:
                    if thinking_active:
                        yield ThinkingFinished()
                    message = self._redacted_text(error)
                    await self._sessions.append(
                        "turn_failed", {"phase": "provider_request", "message": message}
                    )
                    yield AgentFailed(message=f"Provider request failed: {message}")
                    return

                if thinking_active:
                    yield ThinkingFinished()
                elif not thinking_seen_live and response is not None:
                    recovered_thinking = "".join(
                        block.thinking
                        for block in response.blocks
                        if isinstance(block, ThinkingBlock)
                    )
                    has_redacted_thinking = any(
                        isinstance(block, RedactedThinkingBlock) for block in response.blocks
                    )
                    if recovered_thinking or has_redacted_thinking:
                        yield ThinkingStarted()
                        if recovered_thinking:
                            yield ThinkingDelta(text=recovered_thinking)
                        yield ThinkingFinished()

                if (
                    response is not None
                    and response.stop_reason == "max_tokens"
                    and not response.tool_uses
                    and request_attempt < self._response_retry_limit
                ):
                    retry_attempt = request_attempt + 1
                    await self._record_provider_retry(reason="max_tokens", attempt=retry_attempt)
                    yield WarningRaised(
                        "Provider reached the output limit; continuing the response "
                        f"({retry_attempt}/{self._response_retry_limit})"
                    )
                    continuation = ProviderContinuationExchange(
                        response,
                        self._max_tokens_continuation_instruction(),
                    )
                    self._conversation.exchanges.append(continuation)
                    await self._sessions.append("provider_continuation", continuation)
                    request_history = (*request_history, continuation)
                    retry_guidance = None
                    continue
                break

            if response is None:
                message = "Provider response ended without a completed exchange"
                await self._sessions.append("turn_failed", {"message": message})
                yield AgentFailed(message=message)
                return

            await self._sessions.append("assistant_exchange", response)
            if self._context_manager is not None:
                self._context_manager.record_provider_usage(
                    request_history,
                    response.usage,
                    self._supplemental_characters(system_instructions),
                )
            if response.tool_uses:
                results: list[ToolResultBlock] = []
                no_progress_failure: str | None = None
                for call_index, call in enumerate(response.tool_uses):
                    verdict = (
                        self._shell_classifier(call) if self._shell_classifier is not None else None
                    )
                    if verdict is not None:
                        await self._sessions.append(
                            "shell_command_classified",
                            {
                                "call_id": call.call_id,
                                "tool_name": call.name,
                                "command": self._redacted_text(call.input.get("command", "")),
                                "tier": verdict.tier,
                                "matched_rule": verdict.matched_rule,
                                "escapes_workspace": verdict.escapes_workspace,
                                "reason": verdict.reason,
                            },
                        )
                    approval = self._approval_policy.evaluate(call)
                    if (
                        verdict is not None
                        and verdict.forced_action is not None
                        and verdict.matched_rule is None
                        and approval == "allow"
                    ):
                        guardian_note: str | None = None
                        if self._guardian_enabled and call.name == "shell":
                            command = call.input.get("command")
                            if isinstance(command, str):
                                try:
                                    guardian_note = await self._request_shell_guardian_review(
                                        command
                                    )
                                except Exception:
                                    guardian_note = None
                                await self._sessions.append(
                                    "guardian_reviewed",
                                    {
                                        "call_id": call.call_id,
                                        "note": self._redacted_text(guardian_note)
                                        if guardian_note is not None
                                        else None,
                                        "failed": guardian_note is None,
                                    },
                                )
                        message = (
                            "Ran a flagged command without approval: "
                            f"{self._redacted_text(verdict.reason)}"
                        )
                        if guardian_note:
                            message += f" — {self._redacted_text(guardian_note)}"
                        yield WarningRaised(message)
                    if approval == "ask":
                        request_id = uuid.uuid4().hex
                        pending: asyncio.Future[ApprovalDecision] = (
                            asyncio.get_running_loop().create_future()
                        )
                        self._pending_approvals[request_id] = pending
                        guardian_note: str | None = None
                        if self._guardian_enabled and call.name == "shell":
                            command = call.input.get("command")
                            if isinstance(command, str):
                                try:
                                    guardian_note = await self._request_shell_guardian_review(
                                        command
                                    )
                                except Exception:
                                    guardian_note = None
                                await self._sessions.append(
                                    "guardian_reviewed",
                                    {
                                        "call_id": call.call_id,
                                        "note": self._redacted_text(guardian_note)
                                        if guardian_note is not None
                                        else None,
                                        "failed": guardian_note is None,
                                    },
                                )
                        await self._sessions.append(
                            "approval_requested",
                            {
                                "request_id": request_id,
                                "call_id": call.call_id,
                                "tool_name": call.name,
                                "input": call.input,
                                "guardian_note": guardian_note,
                            },
                        )
                        yield ApprovalRequested(
                            request_id=request_id,
                            call_id=call.call_id,
                            tool_name=call.name,
                            arguments=self._redacted_mapping(call.input),
                            guardian_note=self._redacted_text(guardian_note)
                            if guardian_note is not None
                            else None,
                        )
                        try:
                            decision = await pending
                        except asyncio.CancelledError:
                            await self._sessions.append(
                                "approval_resolved",
                                {"request_id": request_id, "decision": "cancelled"},
                            )
                            results.append(
                                ToolResultBlock(
                                    call.call_id,
                                    "cancelled_during_approval",
                                    True,
                                    {"cancelled": True},
                                )
                            )
                            for remaining in response.tool_uses[call_index + 1 :]:
                                results.append(
                                    ToolResultBlock(
                                        remaining.call_id,
                                        "cancelled_before_execution",
                                        True,
                                        {"cancelled": True},
                                    )
                                )
                            continuation = ToolContinuationExchange(response, tuple(results))
                            self._conversation.exchanges.append(continuation)
                            await self._sessions.append("tool_continuation", continuation)
                            await self._sessions.append("turn_cancelled", {"phase": "approval"})
                            yield AgentCancelled(message="Approval cancelled")
                            return
                        finally:
                            self._pending_approvals.pop(request_id, None)
                        self._approval_policy.remember(call, decision)
                        approval = "deny" if decision == "deny" else "allow"
                    if approval == "deny":
                        result = ToolResultBlock(
                            call.call_id,
                            "Permission denied; the tool was not executed",
                            True,
                            {"denied": True},
                        )
                        result, progress_warning, stop_message = await self._check_no_progress(
                            no_progress, call, result
                        )
                        results.append(result)
                        yield ToolStarted(
                            call_id=call.call_id,
                            tool_name=call.name,
                            arguments=self._redacted_mapping(call.input),
                        )
                        await self._sessions.append(
                            "tool_finished",
                            {
                                "call_id": call.call_id,
                                "tool_name": call.name,
                                "is_error": True,
                                "model_content": result.content,
                                "metadata": result.metadata,
                            },
                        )
                        yield ToolFinished(
                            call_id=call.call_id,
                            tool_name=call.name,
                            is_error=True,
                            content=result.content,
                            metadata=result.metadata,
                        )
                        if progress_warning is not None:
                            yield progress_warning
                        no_progress_failure = no_progress_failure or stop_message
                        if stop_message is not None:
                            results.extend(
                                self._no_progress_skipped_results(
                                    response.tool_uses[call_index + 1 :]
                                )
                            )
                            break
                        continue
                    await self._sessions.append(
                        "tool_started",
                        {
                            "call_id": call.call_id,
                            "tool_name": call.name,
                            "input": call.input,
                        },
                    )
                    yield ToolStarted(
                        call_id=call.call_id,
                        tool_name=call.name,
                        arguments=self._redacted_mapping(call.input),
                    )
                    try:
                        result = await self._dispatcher.execute(call)
                    except asyncio.CancelledError:
                        await self._sessions.append(
                            "tool_cancelled",
                            {"call_id": call.call_id, "tool_name": call.name},
                        )
                        results.append(
                            ToolResultBlock(
                                call.call_id,
                                "cancelled_by_user",
                                True,
                                {"cancelled": True},
                            )
                        )
                        for remaining in response.tool_uses[call_index + 1 :]:
                            results.append(
                                ToolResultBlock(
                                    remaining.call_id,
                                    "cancelled_before_execution",
                                    True,
                                    {"cancelled": True},
                                )
                            )
                        continuation = ToolContinuationExchange(response, tuple(results))
                        self._conversation.exchanges.append(continuation)
                        await self._sessions.append("tool_continuation", continuation)
                        await self._sessions.append("turn_cancelled", {"phase": "tool_execution"})
                        yield ToolFinished(
                            call_id=call.call_id,
                            tool_name=call.name,
                            is_error=True,
                            content="cancelled_by_user",
                            metadata={"cancelled": True},
                        )
                        yield AgentCancelled(message="Tool execution cancelled")
                        return
                    except Exception as error:
                        result = ToolResultBlock(
                            call.call_id,
                            self._redacted_text(error),
                            True,
                            {"unexpected_error": True},
                        )
                    result, progress_warning, stop_message = await self._check_no_progress(
                        no_progress, call, result
                    )
                    results.append(result)
                    await self._sessions.append(
                        "tool_finished",
                        {
                            "call_id": call.call_id,
                            "tool_name": call.name,
                            "is_error": result.is_error,
                            "model_content": result.content,
                            "metadata": result.metadata,
                        },
                    )
                    yield ToolFinished(
                        call_id=call.call_id,
                        tool_name=call.name,
                        is_error=result.is_error,
                        content=self._redacted_text(result.content),
                        metadata=self._redacted_mapping(result.metadata),
                    )
                    if progress_warning is not None:
                        yield progress_warning
                    no_progress_failure = no_progress_failure or stop_message
                    if call.name == "activate_skill" and not result.is_error:
                        activated_name = result.metadata.get("skill_name")
                        activated = (
                            self._skills.get(activated_name)
                            if isinstance(activated_name, str)
                            else None
                        )
                        if activated is not None:
                            active_skill = activated
                            await self._sessions.append(
                                "skill_invoked",
                                {
                                    "name": activated.name,
                                    "source": activated.source,
                                    "task": prompt,
                                    "activation": "model",
                                },
                            )
                    if stop_message is not None:
                        results.extend(
                            self._no_progress_skipped_results(response.tool_uses[call_index + 1 :])
                        )
                        break
                continuation = ToolContinuationExchange(
                    assistant=response,
                    results=tuple(results),
                )
                self._conversation.exchanges.append(continuation)
                await self._sessions.append("tool_continuation", continuation)
                if no_progress_failure is not None:
                    await self._sessions.append("turn_failed", {"message": no_progress_failure})
                    yield AgentFailed(message=no_progress_failure)
                    return
                continue

            self._conversation.exchanges.append(response)
            context_update = self._context_usage_changed(system_instructions)
            if context_update is not None:
                yield context_update
            if response.stop_reason == "end_turn":
                await self._sessions.append("turn_completed", {})
                yield AgentCompleted(text=response.text)
                return
            message = f"Provider stopped with reason: {response.stop_reason}"
            if response.stop_reason == "max_tokens" and self._response_retry_limit:
                message = self._retry_exhausted_message(message)
            await self._sessions.append("turn_failed", {"message": message})
            yield AgentFailed(message=message)
            return

        message = "Maximum agent steps reached"
        await self._sessions.append("turn_failed", {"message": message})
        yield AgentFailed(message=message)

    def _context_usage_changed(self, system_instructions: str | None) -> ContextUsageChanged | None:
        if self._context_manager is None:
            return None
        status = self._context_manager.status(
            self._conversation.snapshot(),
            self._supplemental_characters(system_instructions),
        )
        return ContextUsageChanged(
            used_tokens=status.used_tokens,
            context_window=status.context_window,
            level=status.level,
        )

    async def _record_provider_retry(self, *, reason: str, attempt: int) -> None:
        await self._sessions.append(
            "provider_retry",
            {
                "reason": reason,
                "attempt": attempt,
                "max_attempts": self._response_retry_limit,
            },
        )

    def _retry_exhausted_message(self, message: str) -> str:
        return f"{message} after {self._response_retry_limit} retries"

    @staticmethod
    def _with_retry_guidance(base: str | None, guidance: str | None) -> str | None:
        if guidance is None:
            return base
        return f"{base}\n\n{guidance}" if base else guidance

    @staticmethod
    def _protocol_retry_guidance(error_message: str) -> str:
        return (
            "<provider-retry>\n"
            f"The previous response was discarded because it contained {error_message}. "
            "Regenerate the complete response and ensure every tool input is one valid "
            "JSON object. Do not continue a partial tool call.\n"
            "</provider-retry>"
        )

    @staticmethod
    def _max_tokens_continuation_instruction() -> str:
        return (
            "<provider-continuation>\n"
            "The previous response reached the output limit. Continue from exactly where it "
            "stopped. Do not repeat the prior analysis. Complete the current reasoning, and "
            "prefer a valid tool call over additional explanation as soon as you are ready.\n"
            "</provider-continuation>"
        )

    def _redacted_text(self, value: object) -> str:
        redacted = self._display_redactor(str(value))
        return redacted if isinstance(redacted, str) else "[unavailable]"

    def _redacted_mapping(self, value: object) -> dict[str, object]:
        redacted = self._display_redactor(value)
        if not isinstance(redacted, dict):
            return {}
        return {str(key): child for key, child in redacted.items()}

    async def _check_no_progress(
        self,
        detector: NoProgressDetector,
        call: ToolUseBlock,
        result: ToolResultBlock,
    ) -> tuple[ToolResultBlock, WarningRaised | None, str | None]:
        observation = detector.observe(call, result)
        payload = {
            "call_id": call.call_id,
            "tool_name": call.name,
            "repetition_count": observation.repetition_count,
            "fingerprint": observation.fingerprint,
        }
        if observation.action == "warn":
            await self._sessions.append("no_progress_warning", payload)
            warned_result = ToolResultBlock(
                result.tool_use_id,
                result.content + NO_PROGRESS_WARNING,
                result.is_error,
                result.metadata,
            )
            return (
                warned_result,
                WarningRaised(f"The same {call.name} call produced the same result twice in a row"),
                None,
            )
        if observation.action == "stop":
            await self._sessions.append("no_progress_stopped", payload)
            message = (
                "Stopped after the same tool call produced the same result "
                f"{observation.repetition_count} times"
            )
            return result, None, message
        return result, None, None

    @staticmethod
    def _no_progress_skipped_results(
        calls: tuple[ToolUseBlock, ...],
    ) -> tuple[ToolResultBlock, ...]:
        return tuple(
            ToolResultBlock(
                call.call_id,
                "skipped_after_no_progress_stop",
                True,
                {"skipped": True},
            )
            for call in calls
        )
