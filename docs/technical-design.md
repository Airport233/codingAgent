# codingAgent 技术文档

## 1. 文档目的

本文档固定 codingAgent MVP 的技术栈、工程约束、协议适配方式、配置格式、测试策略和开发流程。需求范围以 `docs/requirements.md` 为准，模块关系和运行时数据流以 `docs/architecture.md` 为准。

## 2. 技术选型结论

| 类别 | 选型 | 用途 | 选择理由 |
|---|---|---|---|
| 语言 | Python 3.12+ | 全部后端与 CLI | Windows/macOS 支持良好，文件和进程工具实现直接，测试生态成熟 |
| 包与环境 | uv | 虚拟环境、依赖安装、锁文件、命令运行 | 单一工具覆盖常用工作流，`uv.lock` 可复现依赖 |
| 构建后端 | Hatchling | 构建 Python 包 | 配置简单，支持 `src/` 布局和命令入口 |
| 模型 SDK | Anthropic 官方 Python SDK | Anthropic Messages HTTP/SSE 通信 | 允许使用厂商客户端库；只使用底层 Messages/stream 接口 |
| 数据校验 | Pydantic 2 | 配置、工具参数、内部边界对象 | 能生成 JSON Schema，并提供清晰校验错误 |
| CLI 参数 | Typer | 启动参数和非交互命令 | 类型化命令定义，帮助信息清晰 |
| 交互输入 | prompt_toolkit | 多行输入、历史、快捷键 | 适合终端滚动式交互，不强制全屏 TUI |
| 终端渲染 | Rich | Markdown、颜色、工具状态、差异和进度 | Windows Terminal 与 macOS 终端兼容 |
| 用户目录 | platformdirs | 配置、数据、缓存和日志路径 | 避免手写平台路径分支 |
| 测试 | pytest + pytest-asyncio | 单元、异步和集成测试 | 支持 TDD 与异步 Agent 流程 |
| 代码质量 | Ruff + Pyright | 格式、Lint、静态类型检查 | 反馈快，适合 CI |
| CI | GitHub Actions | Windows/macOS 自动验证 | 与公开 GitHub 仓库直接集成 |
| 搜索后端 | ripgrep，Python 回退 | `code_search` | `rg` 快速；缺失时仍可完成基础搜索 |

不采用 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 或任何其他 Agent 框架。不得使用 Anthropic SDK 的 Tool Runner；Agent 循环、工具调度和上下文管理由本项目实现。

## 3. Python 工程布局

```text
codingAgent/
├── pyproject.toml
├── uv.lock
├── README.txt
├── config.example.toml
├── CODING_AGENT.md
├── src/
│   └── coding_agent/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli/
│       ├── core/
│       ├── providers/
│       ├── tools/
│       ├── context/
│       ├── memory/
│       ├── sessions/
│       ├── config/
│       └── security/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── fixtures/
├── scripts/
│   └── probe_provider.py
└── docs/
    ├── requirements.md
    ├── technical-design.md
    ├── architecture.md
    └── adr/
```

Python 包名为 `coding_agent`，安装后的命令为 `coding-agent`。业务代码采用 `src/` 布局，防止测试意外导入仓库根目录中的未安装代码。

## 4. 依赖边界

### 4.1 允许的 SDK 使用范围

Anthropic SDK 只负责：

- 请求鉴权、HTTP 连接和 SSE 事件解码；
- 暴露 Messages API 的类型化请求与响应；
- 提供底层流式事件。

项目自行负责：

- 把 Provider 事件转换为内部事件；
- 聚合文本、thinking 和工具输入增量；
- 维护消息历史及内容块顺序；
- 校验工具参数并执行本地工具；
- 生成 `tool_result` 并继续请求；
- 判断循环终止、重试、取消和失败；
- 计算上下文压力并执行压缩；
- 持久化及恢复会话。

### 4.2 依赖控制

- 生产依赖必须有明确用途，不为单个小函数引入大型库。
- 所有直接依赖写入 `pyproject.toml`，精确解析结果写入 `uv.lock`。
- 新增生产依赖必须在提交说明中写明理由。
- 标准库能够清晰完成的 JSONL、哈希、路径和异步队列功能不引入第三方替代品。

## 5. Provider 技术方案

### 5.1 支持范围

MVP 支持多个 Anthropic Messages 兼容 Provider 配置档案：

- 阿里云百炼 Anthropic 兼容端点；
- 公司内部 Anthropic 兼容端点；
- 为 Anthropic 官方端点保留配置能力，但不作为交付前置条件。

Provider 不支持自动模型发现。每个模型必须在本地配置中声明模型 ID、上下文窗口、thinking 能力、工具能力和可选扩展参数。

### 5.2 请求方式

- 使用 `AsyncAnthropic`。
- 使用 `messages.stream()` 或等价的底层流式 Messages 接口。
- 工具使用客户端自定义工具定义，不使用服务端代码执行、文本编辑、文件或 Bash 工具。
- thinking 扩展字段通过模型配置决定；百炼特有但 SDK 类型未覆盖的字段通过 `extra_body` 透传。
- 超时、重试次数和最大输出 token 均由模型配置提供默认值，并允许 CLI 覆盖非敏感项。

参考：

- [百炼 Anthropic-compatible Messages](https://help.aliyun.com/zh/model-studio/anthropic-api-messages)
- [Anthropic Messages tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)
- [Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)

### 5.3 内部协议转换

Provider 适配器不向核心暴露 SDK 对象。它输出项目自有事件：

```text
ResponseStarted
ContentBlockStarted
TextDelta
ThinkingDelta
ThinkingSignatureDelta
ToolInputDelta
ContentBlockFinished
UsageUpdated
ResponseFinished
ProviderFailed
```

适配器必须保留原始内容块，特别是 `thinking`、`redacted_thinking`、签名和 `tool_use.id`。UI 可以隐藏 thinking，但不得改变存储及后续请求使用的原始数据。

### 5.4 Provider 能力探测

`scripts/probe_provider.py` 是显式运行的诊断脚本，不进入普通测试。它使用用户本地配置检查：

1. 基础文本请求；
2. 文本流式事件；
3. thinking 的请求字段和返回块；
4. 单个客户端工具调用；
5. 多个客户端工具调用；
6. `tool_result.is_error`；
7. usage 字段；
8. `stop_reason` 值；
9. 百炼扩展 `output_config.effort`；
10. 取消流式请求后的表现。

探测报告必须脱敏，不打印 Base URL、API key、请求头或公司模型的敏感元数据。

## 6. 配置设计

### 6.1 配置位置

用户级配置由 `platformdirs.user_config_dir("codingAgent")` 定位。敏感 Base URL 只允许出现在用户级配置中；API key 推荐仅通过环境变量提供。

仓库提交 `config.example.toml`，其中全部地址和密钥名称均为虚构示例。

### 6.2 配置层级

从低到高：

1. 代码内置默认值；
2. 用户级 `config.toml`；
3. 非敏感项目配置；
4. CLI 参数。

用户级 Provider 配置不允许被项目文件读取、打印或复制到会话记录。项目配置不得声明真实 Base URL 或凭据。

### 6.3 配置示例

```toml
[general]
default_model = "bailian/deepseek-v4-pro"
max_agent_steps = 40
auto_compact_ratio = 0.80
hard_context_ratio = 0.92
show_thinking = false

[providers.bailian]
protocol = "anthropic-messages"
base_url = "https://example.invalid/apps/anthropic"
api_key_env = "DASHSCOPE_API_KEY"
auth_mode = "x-api-key"

[providers.bailian.models.deepseek-v4-pro]
context_window = 131072
max_output_tokens = 8192
thinking_mode = "effort"
thinking_effort = "high"
supports_tools = true

[shell]
mode = "auto"
timeout_seconds = 30
max_timeout_seconds = 300
max_output_bytes = 65536
```

示例数字仅为格式说明，真实模型上限必须由用户配置或经官方文档确认，不能凭模型名称猜测。

### 6.4 MCP Client 扩展预留

MVP 不实现 MCP 传输，也不开放 MCP 端口。工具目录使用 `ToolSource` 抽象获取工具定义：首版只提供 `BuiltinToolSource`，未来可增加 `McpToolSource` 而不修改 AgentLoop 或 ToolDispatcher。

未来首期优先支持本地 `stdio` MCP Server，由 codingAgent 启动并监督子进程。工具对模型暴露为 `mcp__<server>__<tool>`，并复用现有的校验、策略、取消、记录和脱敏链路。Streamable HTTP 以及 codingAgent 作为 MCP Server 均不在 MVP 范围。

## 7. CLI 技术方案

### 7.1 交互形态

首版采用终端滚动式 CLI，不使用全屏 alternate screen：

- Typer 解析启动参数；
- prompt_toolkit 提供多行输入、历史和快捷键；
- Rich 渲染 Markdown、状态、工具调用和错误；
- 核心事件通过异步队列交给 CLI 渲染器。

CLI 不解析 Provider 协议、不执行工具、不修改会话数据。它只提交用户命令并消费核心事件。

### 7.2 取消语义

- 空闲时 `Ctrl+C` 清空当前输入或提示退出。
- 模型流式响应期间 `Ctrl+C` 取消当前 Provider 请求。
- 工具执行期间 `Ctrl+C` 请求 Tool Dispatcher 取消当前工具。
- 会话事件先记录取消原因，再结束当前轮次。
- 连续退出不得留下无法恢复的半个 Exchange。

## 8. 文件与搜索技术方案

### 8.1 路径处理

- 所有文件工具先将路径相对工作区解析为绝对规范路径。
- 在 `Path.resolve()` 后验证目标仍位于工作区。
- 写入不存在文件时同时解析并验证最近存在父目录，防止符号链接逃逸。
- `.git`、凭据文件和用户配置目录至少禁止写入；读取限制单独配置，不能用写入黑名单替代真实路径检查。
- Windows 路径大小写和盘符比较使用平台一致的规范化方式测试。

### 8.2 编辑策略

首版 `edit_file` 采用“范围 + 前置内容校验”：

```text
path
start_line
end_line
expected_content
replacement
expected_file_hash
```

执行前必须满足：

- 文件曾在当前会话读取；
- 当前 SHA-256 与读取时版本一致；
- 指定行范围内容与 `expected_content` 一致。

不满足时拒绝写入，并提示模型重新读取。该方案满足精确行修改，同时避免仅依赖易漂移的行号。唯一文本匹配替换作为后续兼容模式，不进入首版工具 schema，避免过多互斥参数降低模型调用稳定性。

### 8.3 code_search

- 优先通过参数数组启动 `rg`，不拼接 Shell 字符串。
- 支持 literal/regex、glob、大小写和结果上限。
- 回退实现使用 `os.walk`/`Path.rglob` 与 Python `re`，忽略常见大目录。
- 搜索输出按字节和结果数双重截断。

## 9. 跨平台 Shell 技术方案

核心暴露统一 `shell` 工具，具体执行由平台 backend 完成：

### 9.1 Windows

- 默认 executable 为可配置的 PowerShell。
- 使用 Windows 命令行语义，不使用 `shlex.split` 解释 PowerShell。
- 创建独立进程组；优先发送可处理的中断信号，超时后终止进程。
- 进程树清理能力通过独立接口封装并单独测试；不能把 POSIX `killpg` 用作 Windows 实现。
- 保留 Windows 必需环境变量，如 `SystemRoot`、`PATH`、`TEMP`、`TMP`。

### 9.2 macOS

- 默认使用用户 shell；配置可固定为 `/bin/zsh` 或 `/bin/bash`。
- 创建独立 session/process group。
- 取消时先发送 `SIGTERM`，等待宽限期后发送 `SIGKILL`。

### 9.3 共同要求

- `cwd` 必须是工作区内目录。
- 环境变量采用允许列表加显式删除敏感变量，避免 API key 被 `env` 输出。
- stdout/stderr 并发读取并执行流式字节上限，不能先无限读入内存。
- 返回退出码、时长、是否超时、是否取消、截断状态和原始字节数。
- 首版不是 OS 沙箱；自动模式只表示无需逐次审批，不表示命令受到完整系统隔离。

## 10. 上下文与 token 技术方案

- 精确值优先使用 Provider 返回的 usage。
- 请求前用本地估算器计算文本、工具 schema 和历史压力。
- 估算器分别统计 ASCII、CJK、JSON 和内容块开销，再用实际 usage 更新每个 Provider/模型的校准系数。
- 不使用 tiktoken 假装精确计算 DeepSeek 或内部模型 token。
- 自动压缩阈值和硬保护阈值按模型配置。
- Provider 报上下文超限时，执行一次强制压缩并最多重试一次。

压缩不得依赖百炼或 Anthropic 的服务端上下文管理功能。

## 11. 会话存储技术方案

- 使用用户数据目录下的追加式 JSONL。
- 一行一个带 `schema_version`、`event_id`、`session_id`、时间戳和 payload 的事件。
- 写入使用单 writer 队列；每个关键边界 flush。
- 原始 Provider 内容块是审计真相；有效上下文可由压缩 checkpoint 重建。
- 读取时忽略未知可选字段；末尾损坏行可跳过并警告。
- 大工具输出在进入模型和会话前执行同一截断策略，记录原始大小。
- 日志与会话写入前通过统一 Redactor 脱敏。

## 12. 测试策略

### 12.1 测试层次

- 单元测试：纯函数、模型、状态机和错误分类。
- 集成测试：使用 FakeProvider 串起 Agent 循环与真实临时文件工具。
- 契约测试：用户显式运行，访问百炼或公司端点；默认 CI 不运行。
- CLI 测试：输入命令、取消和事件渲染的最小行为，不做脆弱的全屏快照。

### 12.2 P0 测试清单

1. Anthropic 文本/thinking/tool input 流事件聚合；
2. thinking 与签名原样 round-trip；
3. 多个 `tool_use.id` 与 `tool_result.tool_use_id` 完整配对；
4. 非法工具参数通过 `is_error=true` 返回并由模型自愈；
5. 中断恢复补齐未完成工具结果；
6. 路径 `..` 与符号链接逃逸被拒绝；
7. 空文件、递归目录和范围编辑；
8. 文件读后被外部修改时拒绝 edit；
9. Windows/macOS Shell backend 的启动参数和取消路径；
10. Shell 超时与输出截断；
11. 自动和手动压缩使用相同状态机；
12. 压缩失败不替换原上下文；
13. JSONL 尾部损坏恢复；
14. 模型切换后有效上下文移除旧 thinking；
15. Provider、会话和日志脱敏。

### 12.3 CI 矩阵

- `windows-latest` + Python 3.12；
- `macos-latest` + Python 3.12；
- `uv sync --locked`；
- `ruff format --check`；
- `ruff check`；
- `pyright`；
- `pytest` + `pytest-cov`，开启 branch coverage；
- 总语句覆盖率门槛 85%；
- 总分支覆盖率门槛 75%。

测试不得需要真实 API key。

覆盖率计算包含 `src/coding_agent`，排除测试代码、类型检查专用分支和生成文件。CI 应同时生成终端摘要和 XML 报告。不允许为了合并单个功能而临时降低门槛；如需调整，必须先修改需求和技术文档并说明原因。

pytest-cov 负责收集 statement/branch 数据并输出 coverage JSON/XML；仓库内的小型检查脚本分别验证 85% 语句门槛和 75% 分支门槛，避免把两者合并成一个含义不清的总数。

## 13. TDD 与分支流程

总体采用“薄纵切行走骨架 + 分层 TDD”：先写一个失败的端到端验收测试，打通 CLI/Application API、FakeProvider、最小 AgentLoop、一个工具和会话存储；随后在每个纵向功能中，从对外契约向内写单元测试并实现最小代码。不先单独造完所有底层组件，也不用大量 Mock 代替关键集成路径。

每个主体模块使用独立短生命周期分支：

```text
feature/project-scaffold
feature/walking-skeleton
feature/provider-anthropic
feature/tools-read-search
feature/tools-write-edit
feature/shell
feature/sessions
feature/memory
feature/context-compaction
feature/cli-hardening
```

每个分支遵循：

1. 根据需求编号写失败测试；
2. 提交测试或测试与最小骨架；
3. 实现最小代码使测试通过；
4. 重构并保持测试通过；
5. 运行完整质量门禁；
6. 普通 merge 回 `main`，不 squash、不 rebase 已推送历史。

`main` 应始终可安装、可启动且测试通过。并行分支只用于文件和依赖边界明确、不互相修改公共核心类型的模块。

详细的纵切顺序、分支准入条件和里程碑见 `docs/implementation-plan.md`。

## 14. 参考项目使用原则

- OpenCode 与 Codex 仅用于研究模块边界、事件模型、工具调度、压缩和存储思想。
- 不复制与项目规模不相称的 Provider、插件、MCP、多 Agent 或沙箱体系；MCP 在 MVP 中仅保留工具来源扩展边界。
- 引用具体实现时使用固定 commit 链接，并检查许可证。
- Anthropic 协议字段以 Anthropic/百炼官方文档、SDK 源码和 Provider 探测结果为准。

## 15. 技术完成标准

技术基线完成需要同时满足：

- `uv sync --locked` 可在 Windows 与 macOS 安装；
- CLI 可使用 FakeProvider 完整运行；
- P0 测试无需网络即可通过；
- Provider 契约探测能够独立运行且输出脱敏；
- 核心模块不导入 CLI 渲染模块；
- 项目中不存在 Agent 框架或服务端托管工具依赖。
