# codingAgent

一个小巧、自包含的终端编码助手。它通过 Anthropic 兼容的 Messages API 与模型通信，
为模型提供一组聚焦的工具（读写/编辑文件、搜索、shell、skills），并在全屏 TUI 中
渲染对话，支持语法高亮的 diff、流式思考过程展示和会话持久化。

## 特性

- **全屏 TUI**（基于 Textual）：流式响应、可折叠的思考/工具面板、assistant 回复
  支持 Markdown 渲染。
- **Diff 风格的文件编辑**——`edit_file`/`write_file` 工具调用会渲染成带颜色、
  语法高亮的 diff，默认展开显示。
- **内置工具集**——`read_file`、`write_file`、`edit_file`、`mkdir`、
  `code_search`、`shell`。
- **分级工具审批**——`auto` / `ask` / `deny` 三种模式，可用 `Shift+Tab` 实时切换；
  `shell`、`write_file`、`edit_file`、`mkdir` 默认受保护，`ask` 模式下每次调用会
  在 TUI 里弹出确认框展示具体命令/参数；`shell` 命令还会经过内置风险分级器判断，
  被标记为高风险的命令在 `auto` 模式下会被执行，但附带一条警告提示。
- **会话持久化**——每一轮对话都会持久化写入 JSONL；可用 `/resume` 恢复、列出或
  回放历史会话。
- **上下文自动压缩**——上下文占用逼近阈值时自动摘要旧对话，同时保证 system prompt
  和最近几轮对话完整保留、不会被压缩掉。
- **无进展检测**——如果 agent 重复执行完全相同的工具调用并得到完全相同的结果，
  会先警告，连续三次则强制停止。
- **Agent Skills**——按需发现和加载的工作流（`code-review`、`test-fix`、
  `project-map`、`project-init`，以及任何通过 `npx skills add` 安装的技能），
  支持从内置、用户级、项目级三个目录发现。

## 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)（推荐）或 `pip`
- 一个兼容 Anthropic Messages 接口的 API Key

## 安装

```bash
uv sync
```

这会在项目的虚拟环境中安装 `coding-agent` 命令行工具。

## 配置

codingAgent 从一个 TOML 文件读取 provider 配置：

- macOS：`~/Library/Application Support/codingAgent/config.toml`
- Linux：`~/.config/codingAgent/config.toml`

```toml
[general]
default_model = "anthropic/claude-fable-5"
approval_mode = "ask"          # auto | ask | deny

[providers.anthropic]
protocol = "anthropic-messages"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
auth_mode = "x-api-key"        # x-api-key | bearer

[providers.anthropic.models."claude-fable-5"]
context_window = 200000
max_output_tokens = 8192
supports_tools = true
thinking_mode = "disabled"     # disabled | effort | enabled_budget
```

在环境变量里设置好凭证（变量名要跟 `api_key_env` 一致，也可以用下面表格里的
通用覆盖变量）：

```bash
export ANTHROPIC_API_KEY="sk-..."
```

任何配置项都可以在不改配置文件的情况下临时覆盖：

| 环境变量 | 覆盖的配置项 |
|---|---|
| `CODING_AGENT_MODEL` | `general.default_model` |
| `CODING_AGENT_BASE_URL` | `providers.<name>.base_url` |
| `CODING_AGENT_API_KEY` | provider 配置的 `api_key_env` |
| `CODING_AGENT_APPROVAL_MODE` | `general.approval_mode` |

## 运行

```bash
uv run coding-agent
```

常用参数：

```bash
uv run coding-agent --workspace ~/projects/my-app   # 指定工作目录
uv run coding-agent --model anthropic/claude-fable-5
uv run coding-agent --resume                        # 继续最近一次会话
uv run coding-agent --prompt "explain this repo"    # 单次运行，非交互模式
uv run coding-agent --approval-mode auto            # 跳过手动审批
```

运行 `uv run coding-agent --help` 查看完整参数列表（`--max-steps`、
`--max-tokens`、`--context-window`、`--auto-compact-ratio` 等）。

## 使用 TUI

输入任务后按 `Enter` 提交，`Shift+Enter` 在不提交的情况下换行。

| 快捷键 | 作用 |
|---|---|
| `Enter` | 提交输入 |
| `Shift+Enter` | 在输入框内换行 |
| `Shift+Tab` | 循环切换审批模式（auto → ask → deny） |
| `Ctrl+C` | 取消当前运行的任务；空闲时则复制已选中的文本 |
| `Ctrl+Q` | 退出 |

斜杠命令（输入 `/` 会有实时补全提示）：

| 命令 | 说明 |
|---|---|
| `/help` | 显示可用命令 |
| `/model [provider/model]` | 查看或切换当前模型 |
| `/mode [auto\|ask\|deny]` | 查看或设置审批模式 |
| `/skills` | 列出可用的编码工作流 |
| `/skill <name> <task>` | 用指定 skill 执行一个任务 |
| `/context` | 查看上下文占用情况 |
| `/compact` | 手动压缩对话上下文 |
| `/resume` | 恢复一个已保存的会话 |
| `/clear` | 开始一个全新的空会话 |
| `/exit` | 退出 codingAgent |

## 模型可用的工具

| 工具 | 作用 |
|---|---|
| `read_file` | 读取工作区内的 UTF-8 文件（可指定行范围） |
| `write_file` | 创建或覆盖文件 |
| `edit_file` | 替换此前读取过的某段行范围 |
| `mkdir` | 递归创建目录 |
| `code_search` | 搜索文件内容（基于 ripgrep） |
| `shell` | 执行 shell 命令，受审批策略约束 |
| `activate_skill` / `read_skill_resource` | 发现并加载 Agent Skills |

所有文件和 shell 操作都被限制在解析后的工作区根目录内；`WorkspaceGuard` 会
拒绝任何试图越出工作区的路径。

## 审批模式

- **auto** —— 受保护的工具（默认是 `shell`、`write_file`、`edit_file`、`mkdir`）
  无需确认即可执行；若内置风险分级器判定某条 `shell` 命令为高风险，命令在
  `auto` 模式下会被执行，但附带一条警告提示。
- **ask** —— 每一次受保护的工具调用都会在 TUI 中弹出确认（是/否/始终允许）。
- **deny** —— 受保护的工具永远不会执行。

可以用 `Shift+Tab` 或 `/mode <mode>` 实时切换模式；也可以在 `config.toml` 的
`[permissions.shell_rules]` 里为具体命令配置单独的规则。

## Skills

Skill 是一个独立的工作流目录，包含一个 `SKILL.md` 入口文件（YAML frontmatter +
Markdown 正文），可选带有 `references/`、`scripts/`、`assets/` 子目录。
codingAgent 会按优先级从三个位置发现 skill：

1. 内置（`code-review`、`test-fix`、`project-map`、`project-init`）
2. 用户级（与 `config.toml` 同级的用户数据目录下的 `skills/`）
3. 项目级（工作区内的 `.agents/skills`）

可以通过 `npx skills add` 安装 [Agent Skills](https://github.com/anthropics/skills)
生态里的更多技能（TUI 在尝试安装一个未识别的 skill 时会自动调用它）。

## 开发

```bash
uv run pytest              # 运行测试套件
uv run ruff check src tests
uv run pyright src
```

## 项目结构

```
src/coding_agent/
├── application.py     # 核心 agent 循环：对话轮次、工具、审批、上下文压缩
├── runtime.py           # 配置加载，provider/session/tool 的组装
├── cli.py                 # Typer 入口，非 TUI 场景下的 REPL 兜底
├── tui.py                   # Textual 应用
├── prompts.py                # 基础 system prompt
├── no_progress.py             # 重复工具调用检测
├── context/                     # token 预算与上下文压缩
├── memory/                        # CODING_AGENT.md 项目记忆加载器
├── sessions/                        # JSONL 持久化会话日志
├── skills/                            # skill 的发现、加载、安装
├── tools/                               # 内置工具实现
└── providers/                             # Anthropic Messages API 客户端
```
