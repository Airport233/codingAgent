from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic
from platformdirs import user_config_path, user_data_path

from coding_agent.application import AgentApplication
from coding_agent.approval import ApprovalAction, ApprovalMode, ConfigurableApprovalPolicy
from coding_agent.context import ContextBudget, ContextManager, TokenEstimator
from coding_agent.domain import ToolUseBlock
from coding_agent.memory.loader import ProjectMemoryLoader
from coding_agent.providers.anthropic import AnthropicMessagesProvider
from coding_agent.providers.base import Provider
from coding_agent.providers.config import normalize_sdk_base_url
from coding_agent.sessions.jsonl import JsonlSessionRepository, Redactor
from coding_agent.tools.builtin import BuiltinToolSource
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tools.shell import ShellConfig, ShellRiskVerdict, classify_shell_command
from coding_agent.tools.workspace import WorkspaceGuard

_DEFAULT_GUARDED_TOOLS = frozenset({"shell", "write_file", "edit_file", "mkdir"})


class RuntimeConfigurationError(ValueError):
    """The local runtime configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    workspace: Path
    data_root: Path
    model: str
    model_key: str
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    available_models: tuple[str, ...] = ()
    max_tokens: int = 4096
    max_steps: int = 20
    context_window: int = 200_000
    auto_compact_ratio: float = 0.8
    provider_extra_body: dict[str, object] = field(default_factory=dict, repr=False)
    supports_tools: bool = True
    auth_mode: str = "x-api-key"
    approval_mode: ApprovalMode = "auto"
    guarded_tools: frozenset[str] = _DEFAULT_GUARDED_TOOLS
    guardian_enabled: bool = False
    shell_rules: dict[str, ApprovalAction] = field(default_factory=dict, repr=False)
    max_tokens_override: int | None = field(default=None, repr=False)
    max_steps_override: int | None = field(default=None, repr=False)
    context_window_override: int | None = field(default=None, repr=False)
    auto_compact_ratio_override: float | None = field(default=None, repr=False)
    approval_mode_override: ApprovalMode | None = field(default=None, repr=False)

    @classmethod
    def from_environment(
        cls,
        *,
        workspace: Path,
        model: str | None,
        environ: Mapping[str, str] | None = None,
        data_root: Path | None = None,
        max_tokens: int | None = None,
        max_steps: int | None = None,
        context_window: int | None = None,
        auto_compact_ratio: float | None = None,
        approval_mode: str | None = None,
    ) -> RuntimeSettings:
        return cls.load(
            workspace=workspace,
            model=model,
            environ=environ,
            data_root=data_root,
            config_path=Path("__missing_user_config__"),
            max_tokens=max_tokens,
            max_steps=max_steps,
            context_window=context_window,
            auto_compact_ratio=auto_compact_ratio,
            approval_mode=approval_mode,
        )

    @classmethod
    def load(
        cls,
        *,
        workspace: Path,
        model: str | None,
        environ: Mapping[str, str] | None = None,
        data_root: Path | None = None,
        config_path: Path | None = None,
        max_tokens: int | None = None,
        max_steps: int | None = None,
        context_window: int | None = None,
        auto_compact_ratio: float | None = None,
        approval_mode: str | None = None,
    ) -> RuntimeSettings:
        values = os.environ if environ is None else environ
        config = _read_config(config_path or user_config_path("codingAgent") / "config.toml")
        general = _mapping(config.get("general"))
        selected_model = model or values.get("CODING_AGENT_MODEL") or general.get("default_model")
        if not isinstance(selected_model, str) or not selected_model.strip():
            raise RuntimeConfigurationError(
                "Missing model: pass --model, set CODING_AGENT_MODEL, "
                "or configure general.default_model"
            )
        selected_model = selected_model.strip()

        provider = _select_provider(config, selected_model)
        model_id = selected_model.split("/", 1)[-1]
        model_key = _canonical_model_key(config, selected_model)
        model_config = _select_model_config(provider, model_id)
        resolved_max_tokens = _configured_int(
            max_tokens, model_config.get("max_output_tokens"), 4096, "max output tokens"
        )
        resolved_max_steps = _configured_int(
            max_steps, general.get("max_agent_steps"), 20, "max agent steps"
        )
        resolved_context_window = _configured_int(
            context_window, model_config.get("context_window"), 200_000, "context window"
        )
        resolved_auto_ratio = _configured_float(
            auto_compact_ratio,
            general.get("auto_compact_ratio"),
            0.8,
            "auto compaction ratio",
        )
        approval_mode_value = (
            approval_mode
            or values.get("CODING_AGENT_APPROVAL_MODE")
            or general.get("approval_mode", "auto")
        )
        if approval_mode_value not in {"auto", "ask", "deny"}:
            raise RuntimeConfigurationError("Approval mode must be auto, ask, or deny")
        guarded_tools = _guarded_tools(general)
        guardian_enabled = _configured_bool(
            general.get("guardian_enabled"), False, "guardian enabled"
        )
        shell_rules = _shell_rules(config)
        provider_extra_body = _thinking_options(model_config, resolved_max_tokens)
        supports_tools = _configured_bool(
            model_config.get("supports_tools"), True, "supports tools"
        )
        configured_url = values.get("CODING_AGENT_BASE_URL") or provider.get("base_url")
        if not isinstance(configured_url, str) or not configured_url.strip():
            raise RuntimeConfigurationError("Missing Provider Base URL")
        key_environment = provider.get("api_key_env", "CODING_AGENT_API_KEY")
        if not isinstance(key_environment, str):
            raise RuntimeConfigurationError("Provider api_key_env must be a string")
        credential = values.get("CODING_AGENT_API_KEY") or values.get(key_environment)
        if not credential:
            raise RuntimeConfigurationError(
                "Missing API key environment variable: CODING_AGENT_API_KEY"
            )
        auth_mode = provider.get("auth_mode", "x-api-key")
        if not isinstance(auth_mode, str) or auth_mode not in {"x-api-key", "bearer"}:
            raise RuntimeConfigurationError("Provider auth_mode must be x-api-key or bearer")
        if (
            resolved_max_tokens <= 0
            or resolved_max_steps <= 0
            or resolved_context_window <= resolved_max_tokens
        ):
            raise RuntimeConfigurationError("Token and step limits must be positive")
        if not 0 < resolved_auto_ratio < 1:
            raise RuntimeConfigurationError("Auto compaction ratio must be between zero and one")
        resolved_workspace = workspace.resolve()
        if not resolved_workspace.is_dir():
            raise RuntimeConfigurationError("Workspace must be an existing directory")
        try:
            normalize_sdk_base_url(configured_url)
        except ValueError as error:
            raise RuntimeConfigurationError(str(error)) from error
        selected_data_root = (data_root or user_data_path("codingAgent")).resolve()
        return cls(
            workspace=resolved_workspace,
            data_root=selected_data_root,
            model=model_id,
            model_key=model_key,
            base_url=configured_url.strip(),
            api_key=credential,
            available_models=_available_model_keys(config, model_key),
            max_tokens=resolved_max_tokens,
            max_steps=resolved_max_steps,
            context_window=resolved_context_window,
            auto_compact_ratio=resolved_auto_ratio,
            provider_extra_body=provider_extra_body,
            supports_tools=supports_tools,
            auth_mode=auth_mode,
            approval_mode=cast(ApprovalMode, approval_mode_value),
            guarded_tools=guarded_tools,
            guardian_enabled=guardian_enabled,
            shell_rules=shell_rules,
            max_tokens_override=max_tokens,
            max_steps_override=max_steps,
            context_window_override=context_window,
            auto_compact_ratio_override=auto_compact_ratio,
            approval_mode_override=(
                cast(ApprovalMode, approval_mode) if approval_mode is not None else None
            ),
        )

    @property
    def sdk_base_url(self) -> str:
        return normalize_sdk_base_url(self.base_url)


@dataclass(slots=True)
class AgentRuntime:
    application: AgentApplication
    session_id: str
    _client: AsyncAnthropic | None = field(default=None, repr=False)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def __aenter__(self) -> AgentRuntime:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


async def create_runtime(
    settings: RuntimeSettings,
    *,
    resume: bool = False,
    resume_session_id: str | None = None,
    provider: Provider | None = None,
) -> AgentRuntime:
    project_root = discover_project_root(settings.workspace)
    workspace_root = settings.workspace.resolve()
    redactor = Redactor((settings.api_key, settings.base_url, settings.sdk_base_url))
    sessions = JsonlSessionRepository(
        settings.data_root,
        redactor=redactor,
    )
    recovered = (
        await sessions.resume(workspace_root, resume_session_id)
        if resume_session_id is not None
        else await sessions.resume_latest(workspace_root)
        if resume
        else None
    )
    if (resume or resume_session_id is not None) and recovered is None:
        raise RuntimeConfigurationError("No session exists for this project")
    if recovered is None:
        store = sessions.deferred_create(
            workspace_root,
            initial_events=(("model_changed", {"previous": None, "current": settings.model_key}),),
        )
        initial_exchanges = ()
    else:
        store = recovered.store
        initial_exchanges = recovered.conversation

    previous_model = recovered.model if recovered is not None else None
    model_changed = previous_model is not None and previous_model != settings.model_key
    excluded_thinking_indices = (
        frozenset(
            index
            for index, exchange_model in enumerate(recovered.conversation_models)
            if exchange_model is not None and exchange_model != settings.model_key
        )
        if recovered is not None
        else frozenset()
    )
    if recovered is not None and previous_model != settings.model_key:
        await store.append(
            "model_changed",
            {"previous": previous_model, "current": settings.model_key},
        )

    client: AsyncAnthropic | None = None
    selected_provider = provider
    if selected_provider is None:
        credential = settings.api_key
        client_options: dict[str, Any] = {
            "base_url": settings.sdk_base_url,
            "max_retries": 2,
        }
        if settings.auth_mode == "bearer":
            client_options["auth_token"] = credential
        else:
            client_options["api_key"] = credential
        client = AsyncAnthropic(**client_options)
        selected_provider = AnthropicMessagesProvider(
            client=client,
            model=settings.model,
            max_tokens=settings.max_tokens,
            extra_body=settings.provider_extra_body,
            supports_tools=settings.supports_tools,
        )

    shell_guard = WorkspaceGuard(workspace_root)

    def classify_tool_call(call: ToolUseBlock) -> ShellRiskVerdict | None:
        if call.name != "shell":
            return None
        command = call.input.get("command")
        if not isinstance(command, str):
            return None
        return classify_shell_command(command, shell_guard, settings.shell_rules)

    def approval_override(call: ToolUseBlock) -> ApprovalAction | None:
        verdict = classify_tool_call(call)
        if verdict is None or verdict.matched_rule is None:
            # Built-in heuristics (workspace escape, destructive-command patterns)
            # never override the approval decision -- auto must never be escalated
            # into a prompt, and deny must never be relaxed into one. Only an
            # explicit, user-authored rule in [permissions.shell_rules] may
            # override approval_mode in either direction.
            return None
        return verdict.forced_action

    catalog = await ToolCatalog.create(
        (
            BuiltinToolSource(
                workspace_root,
                shell_config=ShellConfig(
                    mode=settings.approval_mode, shell_rules=settings.shell_rules
                ),
            ),
        )
    )
    context_manager = ContextManager(
        ContextBudget(
            context_window=settings.context_window,
            max_output_tokens=settings.max_tokens,
            auto_ratio=settings.auto_compact_ratio,
        ),
        TokenEstimator(),
        excluded_thinking_indices=excluded_thinking_indices,
    )
    if recovered is not None and recovered.compaction is not None:
        context_manager.restore(initial_exchanges, recovered.compaction)
    application = AgentApplication(
        selected_provider,
        ToolDispatcher(catalog),
        store,
        max_steps=settings.max_steps,
        memory_loader=ProjectMemoryLoader(project_root, settings.workspace),
        initial_exchanges=initial_exchanges,
        context_manager=context_manager,
        context_reprojected=model_changed,
        display_redactor=redactor.redact,
        initial_compactions=recovered.compactions if recovered is not None else (),
        approval_policy=ConfigurableApprovalPolicy(
            mode=settings.approval_mode,
            guarded_tools=settings.guarded_tools,
            classify=approval_override,
        ),
        shell_classifier=classify_tool_call,
        guardian_enabled=settings.guardian_enabled,
    )
    return AgentRuntime(application, store.session_id, client)


def discover_project_root(workspace: Path) -> Path:
    resolved = workspace.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return resolved


def _read_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            parsed = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeConfigurationError("Unable to read user configuration") from error
    return parsed


def _select_provider(config: Mapping[str, object], model: str) -> dict[str, object]:
    providers = _mapping(config.get("providers"))
    if not providers:
        return {}
    provider_name = model.split("/", 1)[0] if "/" in model else None
    if provider_name is not None:
        return _mapping(providers.get(provider_name))
    matches = [
        _mapping(value)
        for value in providers.values()
        if model in _model_names(_mapping(value).get("models"))
    ]
    if len(matches) == 1:
        return matches[0]
    if len(providers) == 1:
        return _mapping(next(iter(providers.values())))
    return {}


def _canonical_model_key(config: Mapping[str, object], model: str) -> str:
    if "/" in model:
        return model
    providers = _mapping(config.get("providers"))
    matches = [
        name
        for name, value in providers.items()
        if model in _model_names(_mapping(value).get("models"))
    ]
    if len(matches) == 1:
        return f"{matches[0]}/{model}"
    if len(providers) == 1:
        return f"{next(iter(providers))}/{model}"
    return f"default/{model}"


def _select_model_config(provider: Mapping[str, object], model_id: str) -> dict[str, object]:
    models = provider.get("models")
    if not isinstance(models, Mapping):
        return {}
    return _mapping(models.get(model_id))


def _available_model_keys(config: Mapping[str, object], selected_model: str) -> tuple[str, ...]:
    configured = []
    for provider_name, raw_provider in _mapping(config.get("providers")).items():
        for model_name in _model_names(_mapping(raw_provider).get("models")):
            configured.append(f"{provider_name}/{model_name}")
    return tuple(dict.fromkeys((*configured, selected_model)))


def _configured_int(explicit: int | None, configured: object, default: int, label: str) -> int:
    value = explicit if explicit is not None else configured if configured is not None else default
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeConfigurationError(f"Configured {label} has an invalid type")
    return value


def _configured_float(
    explicit: float | None, configured: object, default: float, label: str
) -> float:
    value = explicit if explicit is not None else configured if configured is not None else default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigurationError(f"Configured {label} has an invalid type")
    return float(value)


def _configured_bool(configured: object, default: bool, label: str) -> bool:
    value = default if configured is None else configured
    if not isinstance(value, bool):
        raise RuntimeConfigurationError(f"Configured {label} has an invalid type")
    return value


def _guarded_tools(general: Mapping[str, object]) -> frozenset[str]:
    configured = general.get("guarded_tools")
    if configured is None:
        return _DEFAULT_GUARDED_TOOLS
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise RuntimeConfigurationError("Configured guarded_tools must be a list of strings")
    return frozenset(configured)


def _shell_rules(config: Mapping[str, object]) -> dict[str, ApprovalAction]:
    permissions = _mapping(config.get("permissions"))
    raw_rules = _mapping(permissions.get("shell_rules"))
    rules: dict[str, ApprovalAction] = {}
    for pattern, action in raw_rules.items():
        if action not in {"allow", "ask", "deny"}:
            raise RuntimeConfigurationError(
                f"Configured shell rule for {pattern!r} must be allow, ask, or deny"
            )
        rules[pattern] = cast(ApprovalAction, action)
    return rules


def _thinking_options(model: Mapping[str, object], max_tokens: int) -> dict[str, object]:
    mode = model.get("thinking_mode", "disabled")
    if mode == "disabled":
        return {}
    if mode == "effort":
        effort = model.get("thinking_effort", "high")
        if effort not in {"low", "medium", "high"}:
            raise RuntimeConfigurationError("Configured thinking effort is invalid")
        return {"output_config": {"effort": effort}}
    if mode == "enabled_budget":
        budget = model.get("thinking_budget")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise RuntimeConfigurationError("Configured thinking budget is invalid")
        if budget >= max_tokens:
            raise RuntimeConfigurationError("Thinking budget must be lower than max output tokens")
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    raise RuntimeConfigurationError("Configured thinking mode is invalid")


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _model_names(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(key for key in value if isinstance(key, str))
    return _string_list(value)
