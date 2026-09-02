codingAgent

Git 仓库：
https://github.com/Airport233/codingAgent

项目简介：
codingAgent 是一个使用 Python 独立实现的终端编程智能体，通过 Anthropic 兼容的 Messages API 与模型交互，能够自主读写文件、搜索代码、执行命令并完成编程任务。
运行方法：
1. 安装 Python 3.12 和 uv。
2. 参考 config.example.toml 创建用户配置，通过环境变量提供 API Key。
3. 在仓库目录执行：
   uv sync
   uv run coding-agent --workspace <目标项目目录>

特色功能：
1. 基于 Textual 的全屏 CLI，支持流式 Thinking、工具卡片、语法高亮 Diff和 markdown渲染。
2. 内置 read_file、write_file、edit_file、mkdir、code_search 和 shell 工具；支持工具的 Ask、Deny、Auto 审批模式及 agent运行时使用 Shift+Tab 的随时快速切换。`shell` 命令还会经过内置风险分级器判断，被标记为高风险的命令在 `auto` 模式下会被执行，但附带一条警告提示。
3. 使用 JSONL 持久化对话、模型内容块和工具记录；/resume 恢复会话时不会重复执行历史工具。
4. 支持上下文容量可视化、/compact 手动压缩和阈值自动压缩；原始历史完整保留，摘要记录任务目标、关键决策、决策原因及后续工作。
5. 支持 CODING_AGENT.md 项目/目录级记忆。
6. 支持文件夹化、按需发现、渐进式加载的 Agent Skills。
7. 跨平台兼容 Windows 和 macOS，使用 TDD、静态检查、测试和覆盖率门禁保证代码质量。
8. Agent 循环包含无进展检测、Provider 异常有限重试和 max_tokens 自动续写，避免了重复调用的失控或长 Thinking 意外中断。
9. 支持通过 /model 切换模型并重新投影上下文。
