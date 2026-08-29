# codingAgent MVP 实施计划

## 1. 文档定位

需求文档定义“做什么”和可验收的质量门槛，技术文档定义“用什么做”和 CI 执行方式，本计划定义“按什么顺序做”。需求或质量门槛变更时，必须先修改需求/技术文档，不能只改实施计划。

## 2. 开发策略

采用“薄纵切行走骨架 + 分层 TDD + 持续集成”：

1. 先写一个失败的端到端验收测试，覆盖用户输入、FakeProvider 工具请求、本地工具结果、最终回复和会话记录。
2. 实现刚好让该测试通过的领域契约与最小组件，形成可运行的行走骨架。
3. 每个后续功能都以纵向切片进入：先添加验收或契约测试，再向内增加状态机、纯函数和边界适配器的单元测试。
4. 分支保持短小，每个分支通过全部 CI 后普通合并到 `main`；`main` 始终可安装、可运行、可测试。

这不是纯自顶向：关键领域逻辑仍通过单元测试驱动。也不是纯自底向：不在主链路验证前独立造完所有底层模块。

## 3. CI 合并门禁

每个功能分支和 `main` 都必须运行：

- Windows latest / Python 3.12；
- macOS latest / Python 3.12；
- `uv sync --locked`；
- Ruff 格式与 Lint；
- Pyright 类型检查；
- pytest 单元、集成和离线端到端测试；
- 总语句覆盖率 ≥ 85%；
- 总分支覆盖率 ≥ 75%；
- 安装后 CLI smoke test。

真实 Provider 契约测试不使用公共 CI 密钥，由用户在本地显式执行。

## 4. 里程碑与纵向切片

### M0：工程与质量门禁

分支：`feature/project-scaffold`

- `pyproject.toml`、`src/` 和 `tests/` 布局；
- uv lockfile、Ruff、Pyright、pytest-cov；
- Windows/macOS GitHub Actions；
- 可安装的 `coding-agent --help`；
- 覆盖 API key、用户配置、会话和日志的 `.gitignore`；
- 仅使用虚构端点与假密钥名称的 `config.example.toml`；
- 本地与 CI 可执行的敏感信息扫描，从第一个代码分支开始阻止凭据入库；
- 可安装的最小项目通过覆盖率与 smoke test 配置验证。

### M1：行走骨架

分支：`feature/walking-skeleton`

- 领域内容块、CoreEvent 和 Application API；
- 脚本化 FakeProvider；
- 最小 AgentLoop 和 `BuiltinToolSource`；
- `read_file` 与内存会话存储；
- 一个离线端到端测试打通整条链路。

### M2：Anthropic Provider 协议

分支：`feature/provider-anthropic`

- 流式内容块聚合；
- thinking/signature 原样往返；
- tool_use/tool_result 配对与错误结果；
- 脱敏错误、有限重试与本地 Provider probe。

### M3：内置工具组

分支：按边界拆分 `feature/tools-read-search`、`feature/tools-write-edit`、`feature/shell`

- WorkspaceGuard 和 ReadSet；
- 读取、搜索、写入、目录、精确编辑；
- Windows/macOS Shell backend；
- 取消、超时、截断、冲突和越界测试。

### M4：会话、恢复与记忆

分支：`feature/sessions`，随后 `feature/memory`

- JSONL 事件存储与版本化；
- 异常中断恢复与末行损坏容错；
- `CODING_AGENT.md` 项目/目录级记忆；
- 摘要、工具记录和原始协议内容保留。

### M4.5：可运行集成基线

分支：`feature/runtime-integration`

该切片在 M5 之前完成，优先降低交付主流程风险：

- 从环境变量和用户级配置加载 Provider 与默认模型；
- 装配真实 Provider、内置工具、JSONL 会话和项目记忆；
- 将会话恢复结果安装到 Application，避免恢复后重放工具；
- 提供可在真实工作区连续输入任务的最小滚动式 CLI；
- 用 FakeProvider 运行离线端到端测试，并显式运行真实 Provider 冒烟验证。

本切片不实现上下文压缩、完整 Slash 命令集或终端视觉打磨；这些仍分别属于 M5 和 M6。

### M5：上下文与压缩

分支：`feature/context-compaction`

- token 估算、实际 usage 校准与容量显示；
- `/compact` 与自动压缩；
- Exchange 边界、原子安装、失败回滚；
- 切换模型后的 thinking 投影规则。

### M6：CLI 产品化与发布验收

分支：`feature/cli-hardening`

- Textual 全屏 TUI：会话区、输入区、状态栏、工具卡片、斜杠命令和 thinking 折叠；
- `--prompt` 保留稳定的 stdout 非交互模式；
- 取消、恢复、错误呈现和状态区；
- Windows/macOS 干净环境安装与主验收场景；
- 检查所有 P0 需求的测试映射。

## 5. MCP 后续任务

MCP 不得阻塞 M0–M6。主体功能稳定后，可单独建立 `feature/mcp-client-stdio`：

1. 读取用户级 MCP Server 配置；
2. 管理 stdio 子进程生命周期；
3. 完成 initialize、工具发现、调用、取消和错误映射；
4. 通过 `McpToolSource` 注册命名空间工具；
5. 用伪造 MCP Server 进行无网络契约测试；
6. 再选择 CodeGraph 或 OpenCodeReview 做显式本地集成验证。

Streamable HTTP、远程鉴权和 codingAgent MCP Server 不属于该首期任务。

## 6. 每个分支的完成条件

- 对应需求编号已写入测试名、docstring 或测试映射表；
- 能看到至少一次红→绿→重构的提交轨迹；
- 新的错误路径有测试，不只覆盖 happy path；
- 本地全部门禁通过，远程 Windows/macOS CI 通过；
- 未降低覆盖率门槛，未引入真实密钥或敏感端点；
- 普通合并到 `main`，保留完整开发历史。
