# lazydspy

`lazydspy` 是一个 Agent 驱动的 CLI 工具，通过交互式对话了解你的 Prompt 优化需求，并**生成一个可直接运行的单文件 DSPy 优化脚本**。生成的脚本遵循现代 Python 实践（PEP 723 元数据、Pydantic v2 模型、Typer CLI、Rich 输出），支持 **quick/full** 运行模式以及用于长时间运行的 **checkpoint（断点续传）** 功能。

`lazydspy` 本身不执行完整的优化过程——它负责创建脚本和指南，由你使用自己的数据和 API 密钥独立运行。

## 核心特性

- **Agent 优先**：使用 Claude Agent SDK 通过自然对话动态收集需求，而非硬编码的问题列表
- **动态脚本生成**：Agent 根据你的具体需求编写代码，而非模板填充
- **符合 PEP 723**：生成的脚本包含内联元数据，便于依赖管理
- **现代 Python 技术栈**：Pydantic v2 数据验证、Typer CLI、Rich 终端输出
- **双运行模式**：Quick 模式用于低成本探索，Full 模式用于生产级优化
- **Checkpoint 支持**：长时间优化任务的定期检查点，支持断点恢复

## 快速开始

### 前置要求

- Python **3.12+**
- [uv](https://github.com/astral-sh/uv) 包管理器
- Anthropic API 密钥（Claude）

### 安装

```bash
git clone https://github.com/your-org/lazydspy.git
cd lazydspy
uv sync
```

### 基本用法

启动交互式对话以生成脚本：

```bash
uv run lazydspy chat
```

或直接运行（默认执行 `chat`）：

```bash
uv run lazydspy
```

## CLI 选项

```bash
uv run lazydspy chat [OPTIONS]

选项:
  -m, --model TEXT       Claude 模型名称（默认: claude-opus-4.5）
  --base-url TEXT        自定义 API 端点
  --auth-token TEXT      API 令牌（或设置 ANTHROPIC_API_KEY 环境变量）
  --debug                启用调试模式
  -v, --version          显示版本
  --help                 显示帮助信息
```

## 生成的输出

对话完成后，`lazydspy` 会在 `generated/<session_id>/` 下创建一个文件夹：

```
generated/<session_id>/
├── pipeline.py      # 主优化脚本（符合 PEP 723）
├── metadata.json    # 使用的配置信息
├── README.md        # 使用说明
├── DATA_GUIDE.md    # 数据准备指南
└── sample-data/     # 可选的示例 JSONL
    └── train.jsonl
```

### 运行生成的脚本

```bash
# Quick 模式 - 低成本探索
uv run generated/<session_id>/pipeline.py --mode quick

# Full 模式 + checkpoint
uv run generated/<session_id>/pipeline.py --mode full --checkpoint-dir checkpoints --resume
```

## 配置

### 环境变量

| 变量 | 描述 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API 密钥（必需） |
| `ANTHROPIC_MODEL` | 模型名称（默认: `claude-opus-4.5`） |
| `ANTHROPIC_BASE_URL` | 自定义 API 端点（可选） |
| `ANTHROPIC_AUTH_TOKEN` | ANTHROPIC_API_KEY 的替代方式 |
| `OPENAI_API_KEY` | OpenAI API 密钥（如果使用 OpenAI 模型） |
| `LAZYDSPY_DEBUG` | 启用调试模式（`1`, `true`, `yes`） |

### 自定义 Claude 端点

用于本地 Claude 代理或自托管端点：

```bash
# 通过环境变量
export ANTHROPIC_BASE_URL=http://localhost:8030
export ANTHROPIC_AUTH_TOKEN=dev-local-token
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# 或通过 CLI
uv run lazydspy chat --base-url http://localhost:8030 --auth-token dev-local-token --model claude-3-5-sonnet-20241022
```

## 优化器

### GEPA (GEvalPromptedAssembly)

基于进化策略的低成本 Prompt 优化器，适合通用任务。

**适用场景**：摘要、分类、生成、通用任务

| 模式 | breadth | depth | temperature |
|------|---------|-------|-------------|
| quick | 2 | 2 | 0.3 |
| full | 4 | 4 | 0.7 |

### MIPROv2 (Model-based Instruction Prompt Refinement Optimizer v2)

基于模型的指令优化器，更适合复杂推理任务。

**适用场景**：检索、评分、问答、推理

| 模式 | search_size | temperature |
|------|-------------|-------------|
| quick | 8 | 0.3 |
| full | 16 | 0.6 |

### 成本对比

- Quick 模式通常比 Full 模式便宜 5-10 倍
- **建议**：先用 `quick` 验证效果，再用 `full` 进行生产级优化

## 架构

本项目遵循 **Agentic 架构**：

1. **Agent 由 System Prompt 驱动**（`agent/prompts.py`），而非硬编码逻辑
2. **通过 MCP 模式提供工具**（`tools/`），用于文件、数据和领域操作
3. **动态脚本生成**——Agent 编写代码，而非模板填充

```
src/lazydspy/
├── agent/           # 核心 Agent 模块
│   ├── config.py    # AgentConfig
│   ├── prompts.py   # SYSTEM_PROMPT
│   ├── runner.py    # AgentRunner
│   └── session.py   # ConversationSession
│
├── models/          # Pydantic 数据模型
│   ├── config.py    # GenerationConfig
│   └── hyperparams.py
│
├── knowledge/       # 领域知识
│   ├── optimizers.py
│   └── cost_models.py
│
└── tools/           # MCP 工具
    ├── file_ops.py
    ├── data_ops.py
    └── domain_ops.py
```

## 开发

### 代码检查

在仓库根目录运行（推荐顺序）：

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

或一次性运行所有检查：

```bash
make check
```

### 运行测试

```bash
uv run pytest tests/ -v
uv run pytest tests/test_cli_session.py::test_generation_config_overrides_are_typed -v
```

## 依赖

**核心**：
- `anthropic` - Claude API 客户端
- `dspy` - DSPy 框架
- `pydantic>=2` - 数据验证
- `typer>=0.12` - CLI 框架
- `rich` - 终端格式化
- `claude-agent-sdk==0.1.17` - Claude Agent SDK

**开发**：
- `pytest>=8.3` - 测试
- `mypy>=1.11` - 类型检查
- `ruff>=0.6` - 代码检查

## 故障排除

### Windows 下中文输出问题

使用 PowerShell 代替 cmd：

```bash
pwsh -NoProfile -Command "uv run lazydspy chat"
```

### API 令牌未设置

通过环境变量或 CLI 提供：

```bash
export ANTHROPIC_API_KEY=your-key
# 或
uv run lazydspy chat --auth-token your-key
```

## 许可证

MIT
