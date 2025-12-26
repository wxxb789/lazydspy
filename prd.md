# lazydspy – 产品需求文档

> **版本**: 0.2.0  
> **更新日期**: 2025-12-26  
> **状态**: 已实现核心功能

---

## 愿景

**lazydspy** 是一个由 **Claude Agent SDK** 驱动的交互式 CLI Agent，通过对话引导用户生成可直接运行的、单文件 DSPy prompt 优化脚本。该工具本身**不执行**优化——它生成一个独立的 Python 脚本，用户使用自己的数据和 API 密钥独立运行。

---

## 目标

| 目标 | 状态 | 说明 |
|------|------|------|
| 降低门槛 | ✅ 已实现 | 通过对话收集需求，消除样板代码和配置猜测 |
| 生成生产级产物 | ✅ 已实现 | 单个 `.py` 文件，遵循 PEP 723 和现代 Python 实践 |
| 前置成本/复杂度权衡 | ✅ 已实现 | 提供成本估算工具，帮助用户选择 quick/full 模式 |
| 确保可复现性 | ✅ 已实现 | 生成的脚本支持 checkpointing 和确定性数据处理 |

---

## 核心原则

### 1️⃣ Agentic First

**用 System Prompt + Agent Skills 赋能 Agent，而非 hardcode。**

| 层级 | 职责 | 当前实现 |
|------|------|----------|
| **System Prompt** | 定义 Agent 角色、边界、行为准则 | `prompts.py` - preset + append 模式 |
| **Agent Skills** | 封装可复用能力，Agent 按需调用 | `tools.py` - 3 个 MCP 工具 |
| **SDK 内置工具** | 文件操作、Shell 命令等 | Read, Write, Edit, Bash, Glob, Grep |
| **动态对话** | Agent 根据上下文自主决定提问 | 无预设问题列表 |

### 2️⃣ 代码极简

> **追求代码的极致简单、清晰、优雅。**

- 优先使用 **Pydantic v2** 简化验证和序列化
- 单一职责：每个模块只做一件事
- 显式优于隐式：不依赖魔法行为
- 全项目仅 6 个核心源文件，约 600 行代码

### 3️⃣ 动态生成

> **脚本由 Agent 根据需求直接编写，不使用预设模板。**

- ❌ 不硬编码问题列表
- ❌ 不硬编码脚本模板
- ✅ Agent 理解需求后动态生成代码

---

## 当前架构（v0.2.0）

### 目录结构

```
src/lazydspy/
├── __init__.py       # 包导出，延迟加载（43 行）
├── __main__.py       # CLI 入口点（9 行）
├── cli.py            # Typer CLI，参数处理（130 行）
├── agent.py          # Agent 核心，ClaudeSDKClient 多轮对话（167 行）
├── tools.py          # MCP 工具定义，@tool 装饰器（177 行）
├── prompts.py        # System Prompt 配置（83 行）
└── knowledge/        # 领域知识
    ├── __init__.py   # 模块导出（32 行）
    ├── optimizers.py # OptimizerInfo（Pydantic v2），优化器注册表（95 行）
    └── cost_models.py# 成本估算模型（128 行）
```

**总代码量**: ~860 行

### 模块职责

| 模块 | 职责 |
|------|------|
| `cli.py` | Typer CLI，处理命令行参数，验证 API Key |
| `agent.py` | Agent 类，管理 `ClaudeSDKClient` 多轮对话循环 |
| `tools.py` | 定义 MCP 工具，业务逻辑分离为 `*_impl` 函数便于测试 |
| `prompts.py` | System Prompt 配置，使用 `claude_code` 预设 + append |
| `knowledge/` | 领域知识：优化器信息、成本模型、推荐逻辑 |

### 数据流

```
┌──────────────────────────────────────────────────────────────────┐
│  用户输入                                                         │
│     ↓                                                            │
│  cli.py (Typer) → 验证 API Key → 创建 AgentConfig                 │
│     ↓                                                            │
│  agent.py (Agent) → ClaudeSDKClient(options)                     │
│     ↓                                                            │
│  多轮对话循环:                                                    │
│     ├─ 用户输入 → client.query()                                 │
│     ├─ Agent 响应 → client.receive_response()                    │
│     ├─ 工具调用 → MCP 工具 (tools.py) 或 SDK 内置工具             │
│     └─ 循环直到用户退出                                           │
│     ↓                                                            │
│  生成产物: generated/<session_id>/                                │
│     ├─ pipeline.py   (主优化脚本)                                 │
│     ├─ metadata.json (配置信息)                                   │
│     └─ README.md     (使用说明)                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## MCP 工具定义

| 工具名 | 说明 | 参数 |
|--------|------|------|
| `estimate_cost` | 估算 DSPy 优化的 API 调用费用 | optimizer, mode, dataset_size, model |
| `list_optimizers` | 列出所有可用的 DSPy 优化器 | 无 |
| `get_defaults` | 获取优化器的默认配置 | optimizer, mode, scenario |

**工具名格式**: `mcp__lazydspy__<tool_name>`

---

## 用户工作流

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 用户运行 `uv run lazydspy` 或 `uv run lazydspy chat`         │
│                                                                  │
│  2. Agent 通过动态对话了解需求：                                   │
│       • 自主判断需要了解的信息                                     │
│       • 根据用户回答动态调整后续问题                                │
│       • 调用 estimate_cost 帮助用户理解费用                        │
│       • 调用 get_defaults 获取推荐配置                             │
│                                                                  │
│  3. Agent 与用户确认需求摘要                                       │
│                                                                  │
│  4. Agent 使用 SDK 内置工具在 `generated/<session_id>/` 下生成：   │
│       • pipeline.py   – 动态生成的优化脚本                         │
│       • metadata.json – 配置信息                                  │
│       • README.md     – 运行说明                                  │
│                                                                  │
│  5. 用户独立运行生成的脚本：                                        │
│       • quick 模式：快速验证效果                                   │
│       • full 模式：生产级优化 + checkpointing                      │
│                                                                  │
│  6. 脚本将优化后的 prompt 输出到控制台 + 文件                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 生成脚本规范

| 方面 | 要求 |
|------|------|
| **元数据** | PEP 723 内联元数据；声明 `python >=3.12` 及所有依赖 |
| **CLI** | 使用 **Typer** + **Rich**；支持 `--mode`、`--checkpoint-dir`、`--resume` 等 |
| **Checkpointing** | 默认 10–20 个检查点；可通过参数配置 |
| **数据格式** | UTF-8 JSONL，使用 Pydantic v2 验证 |
| **优化器** | 默认使用低成本配置（如 GEPA + quick）；用户可选择 MIPROv2 或 full |
| **输出** | 将优化后的 prompt 打印到控制台**并**持久化到文件 |

### 优化器预设

#### GEPA（低成本，推荐通用场景）

| 模式 | breadth | depth | temperature |
|------|---------|-------|-------------|
| quick | 2 | 2 | 0.3 |
| full | 4 | 4 | 0.7 |

#### MIPROv2（适合复杂推理）

| 模式 | search_size | temperature |
|------|-------------|-------------|
| quick | 8 | 0.3 |
| full | 16 | 0.6 |

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| **Agent 核心** | Claude Agent SDK ≥0.1.17 | 项目驱动核心，提供多轮对话和工具调用 |
| **CLI 框架** | Typer ≥0.12 + Rich | 命令行界面和终端格式化 |
| **数据验证** | Pydantic v2 | 优化器信息等领域模型 |
| **包管理** | uv | 依赖安装和脚本运行 |
| **代码检查** | ruff, mypy --strict | Linting 和类型检查 |
| **测试** | pytest ≥8.3 | 单元测试 |

### 依赖（pyproject.toml）

```toml
dependencies = [
    "claude-agent-sdk>=0.1.17",
    "rich",
    "typer>=0.12",
]
```

---

## 配置与环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_AUTH_TOKEN` | Claude API 令牌（优先级 1） | - |
| `ANTHROPIC_API_KEY` | Claude API 密钥（优先级 2） | - |
| `ANTHROPIC_MODEL` | 模型名称 | `claude-sonnet-4-20250514` |
| `ANTHROPIC_BASE_URL` | 自定义 API 端点 | - |
| `LAZYDSPY_DEBUG` | 启用调试模式 | `false` |

---

## 约束与非目标

- **不是运行时优化器**：lazydspy 生成脚本，不自己运行完整优化
- **无 GUI**：仅 CLI；可使用 Rich 增强 TUI
- **单文件输出**：生成的脚本必须可独立运行（除声明的依赖外无外部模块导入）
- **不自动上传数据集**：用户自己提供数据路径
- **🚫 不硬编码问题列表**：对话流程由 Agent 动态驱动
- **🚫 不硬编码脚本模板**：脚本由 Agent 根据需求直接生成

---

## 成功指标

| 指标 | 状态 | 说明 |
|------|------|------|
| 5 分钟内完成脚本生成 | ✅ | 从 `lazydspy chat` 到获得可运行的 `pipeline.py` |
| 生成脚本通过 ruff + mypy | ✅ | Agent 应遵循 System Prompt 中的规范 |
| 优化后 prompt 清晰呈现 | ✅ | 脚本执行结束时打印并保存 |
| 无硬编码问题列表/模板 | ✅ | 代码库中不存在 |
| 新场景仅需调整 Prompt/Skill | ✅ | 无需修改核心对话逻辑 |
| 领域模型使用 Pydantic v2 | ✅ | `OptimizerInfo` 等 |

---

## 未来规划

### 短期（v0.3.0）

- [ ] **Subagent 支持**：引入专门化的子 Agent 处理特定任务
  - 数据格式分析 Subagent
  - 脚本验证 Subagent
- [ ] **更多优化器**：支持 BootstrapFewShot、COPRO 等
- [ ] **成本追踪**：记录实际 API 调用费用

### 中期

- [ ] **非交互模式**：支持 CI 流水线批量生成
- [ ] **多轮迭代优化**：基于初始脚本结果进行改进
- [ ] **遥测数据**：收集匿名使用数据改进推荐

### 长期

- [ ] **插件系统**：允许用户扩展 Agent 能力
- [ ] **多语言支持**：生成其他语言的优化脚本

---

## 参考资料

- [DSPy 文档](https://dspy.ai)
  - [API 参考](https://dspy.ai/api/)
  - [教程](https://dspy.ai/tutorials/)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
  - [官方文档](https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python)
