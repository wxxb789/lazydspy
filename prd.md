# lazydspy – 产品需求文档

## 项目背景

当前项目是由 **Codex Agent** 实现的半成品，已**完全偏离**原有设计意图：

- **原有设计**：围绕 Agentic 理念，让 Agent 动态驱动对话和代码生成
- **当前状态**：充斥大量 hardcode——固定问题列表、模板字符串、僵化 workflow
- **结论**：需要 **Claude Code 彻底重构，完全重新实现**

> ⚠️ 本次重构不是修补，而是从零开始基于正确的 Agentic 架构重建。

---

## 愿景

**lazydspy** 是一个由 **Claude Agent SDK** 驱动的交互式 CLI Agent，通过对话引导用户生成一个可直接运行的、单文件的 DSPy prompt 优化脚本。该工具本身**不执行**优化——它生成一个独立的 Python 脚本，用户使用自己的数据和 API 密钥独立运行。

---

## 目标

1. **降低门槛**：消除样板代码和配置猜测，让 DSPy prompt 优化更易上手。
2. **生成生产级产物**：单个 `.py` 文件 + 配套文档，遵循现代 Python 最佳实践。
3. **前置成本/复杂度权衡**：让用户在做决策前了解 quick vs. full 模式、优化器选择等影响。
4. **确保可复现性**：通过 checkpointing 和确定性数据处理保证结果可复现。

---

## 核心原则

### 1️⃣ 设计先行

> **总是先设计，再编码。总是从项目整体结构思考，再着手实现。**

- 动手写代码前，先明确架构、模块边界、数据流
- 对复杂功能先画出流程图或伪代码
- 避免"边写边想"导致的结构混乱

### 2️⃣ Agentic First

**用有限的 Prompt + Subagents + Agent Skills 赋能 Agent，而非预设不变的 hardcode。**

| 层级 | 职责 | 示例 |
|------|------|------|
| **System Prompt** | 定义 Agent 的角色、边界、行为准则 | "你是 DSPy 优化脚本生成专家..." |
| **Agent Skills** | 封装可复用的能力模块，Agent 按需调用 | `generate_script`、`validate_schema`、`estimate_cost` |
| **Subagents** | 处理特定子任务的专门化 Agent | 数据格式分析 Agent、优化器选择 Agent |
| **动态对话** | Agent 根据上下文自主决定提问内容和顺序 | 不预设问题列表，根据用户回答动态追问 |

### 3️⃣ 代码极简

> **追求代码的极致简单、清晰、优雅。**

- **总是使用 pydantic v2** 简化验证、序列化、object mapping
- 避免过度抽象，能一眼看懂的代码优于"聪明"的代码
- 单一职责：每个函数/类只做一件事
- 显式优于隐式：不依赖魔法行为

---

## 反模式 vs. 正确方向

### 🚫 反模式（当前问题）

当前项目由 Codex Agent 实现，已严重偏离原有 Agentic 设计：
- 硬编码的问题列表（hardcoded question flow）
- 硬编码的脚本模板（hardcoded script templates）
- 固定的 workflow 流程，缺乏灵活性
- Python 代码承担了本应由 Agent 动态处理的逻辑

### ✅ 正确方向

#### 类 Claude Code 体验

Agent 应具备以下交互能力（通过 Prompt 约束，而非代码控制）：

1. **主动提问**：Agent 自主判断缺失信息并提问，而非按固定顺序询问
2. **边界界定**：识别并拒绝超出范围的需求，引导用户聚焦
3. **范围控制**：提示用户不要把问题设计得过于宏大，建议拆分
4. **确认机制**：关键决策前与用户确认，避免假设
5. **迭代澄清**：对模糊需求反复追问直到清晰

#### Skill vs. Hardcode 判断标准

| 场景 | 处理方式 |
|------|----------|
| 逻辑可由 LLM 自然完成 | 纳入 System Prompt 指导 |
| 需要确定性计算（如成本估算公式） | 封装为 Agent Skill（Tool） |
| 可复用的多步流程 | 封装为 Subagent |
| 需要外部 I/O（文件读写、API 调用） | 封装为 Agent Skill（Tool） |
| ❌ 固定问答流程 | **不应存在** |
| ❌ 硬编码模板字符串 | **不应存在** |

---

## 代码规范（Code Guide）

```python
# ✅ 推荐：使用 pydantic v2 简化一切
from pydantic import BaseModel, Field
from typing import Literal

class OptimizationConfig(BaseModel):
    task_type: str = Field(..., description="任务类型")
    input_fields: list[str] = Field(default_factory=list)
    output_field: str = Field(...)
    optimizer: str = Field(default="gepa")
    mode: Literal["quick", "full"] = Field(default="quick")

# ❌ 避免：手动 dict 操作 + 类型不安全
config = {
    "task_type": data.get("task_type", ""),
    "input_fields": data.get("input_fields") or [],
    # ...
}
```

### 规范要点

| 方面 | 规范 |
|------|------|
| **数据模型** | 全部使用 pydantic v2 `BaseModel`，利用自动验证和序列化 |
| **类型标注** | 100% 类型覆盖，避免 `Any`、`Dict[str, Any]` |
| **函数设计** | 单一职责，参数 ≤5 个，超过则用 pydantic model 封装 |
| **错误处理** | 使用 pydantic `ValidationError`，避免裸 `try/except` |
| **配置管理** | 使用 `pydantic-settings` 处理环境变量和配置文件 |
| **代码风格** | 遵循 `ruff` 规则，通过 `mypy --strict` 检查 |

---

## 用户工作流

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 用户运行 `uv run lazydspy chat`                              │
│  2. Agent 通过动态对话了解需求（非固定问题列表）：                   │
│       • Agent 自主判断需要了解的信息                               │
│       • 根据用户回答动态调整后续问题                                │
│       • 主动识别范围过大的需求并引导收敛                            │
│       • 对模糊点反复追问直到清晰                                   │
│  3. Agent 与用户确认需求摘要                                      │
│  4. Agent 调用 Skills 在 `generated/<session_id>/` 下生成产物：   │
│       • pipeline.py   – 动态生成，非模板填充                       │
│       • metadata.json – 捕获的配置                                │
│       • DATA_GUIDE.md – 数据准备指南                              │
│       • README.md     – 运行说明                                  │
│       • (可选) sample-data/train.jsonl 示例桩                     │
│  5. 用户运行生成的脚本：quick 模式做快速验证，                       │
│     然后 full 模式 + checkpointing 用于生产                       │
│  6. 脚本将优化后的 prompt 输出到控制台 + 文件                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     lazydspy Main Agent                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  System Prompt（有限约束）                                  │  │
│  │  • 角色定义：DSPy 优化脚本生成专家                           │  │
│  │  • 行为准则：主动提问、边界控制、范围收敛                     │  │
│  │  • 输出规范：PEP 723、Typer CLI、pydantic 等                │  │
│  │  • 领域知识：DSPy 优化器特性、成本模型（概要）                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│              ┌───────────────┼───────────────┐                   │
│              ▼               ▼               ▼                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │  Agent Skills   │ │  Agent Skills   │ │  Agent Skills   │    │
│  │  (Tools)        │ │  (Tools)        │ │  (Tools)        │    │
│  ├─────────────────┤ ├─────────────────┤ ├─────────────────┤    │
│  │ write_file      │ │ validate_jsonl  │ │ estimate_cost   │    │
│  │ read_file       │ │ check_schema    │ │ list_optimizers │    │
│  │ create_dir      │ │ sample_data     │ │ get_defaults    │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
│                              │                                   │
│              ┌───────────────┴───────────────┐                   │
│              ▼                               ▼                   │
│  ┌─────────────────────────┐ ┌─────────────────────────────┐    │
│  │  Subagent (可选)         │ │  Subagent (可选)            │    │
│  │  数据格式分析             │ │  脚本验证                    │    │
│  └─────────────────────────┘ └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 生成脚本要求

| 方面                 | 要求                                                                                                |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| **元数据**           | PEP 723 内联元数据；声明 `python >=3.12` 及所有依赖。                                                 |
| **CLI**              | 使用 **Typer** + **Rich** 构建；支持 `--mode quick/full`、`--checkpoint-dir`、`--resume` 等参数。     |
| **Checkpointing**    | 默认 ~10–20 个检查点；可通过 `--checkpoint-interval` 和 `--max-checkpoints` 配置。                    |
| **数据格式**          | UTF-8 JSONL，输入/输出字段名明确，使用 pydantic 验证。                                                |
| **优化器**           | 默认使用最低成本配置（如 GEPA + 小样本）；用户可选择 MIPROv2 或更重的配置。                             |
| **输出**             | 将最终优化后的 prompt 打印到控制台**并**持久化到 `optimized_prompt.txt`。                              |
| **生成方式**         | ⚠️ **动态生成，非模板填充**。Agent 根据需求理解直接编写代码，不使用预设模板。                           |

---

## 技术栈

| 组件                 | 选型                                        |
| -------------------- | ------------------------------------------- |
| **Agent 核心**       | Claude Agent SDK **v0.1.17**（项目驱动核心）  |
| CLI 框架             | **Typer** + **Rich**                        |
| 数据验证             | **pydantic v2**（全面使用）                  |
| 包运行器             | **uv** (`uv run`, `uv sync`)                |
| 代码检查 / 类型检查   | **ruff**, **mypy --strict**                 |
| 测试                 | **pytest**                                  |

---

## 约束与非目标

- **不是运行时优化器**：lazydspy 生成脚本，不自己运行完整优化。
- **无 GUI**：仅 CLI；可使用 Rich 增强 TUI。
- **单文件输出**：生成的脚本必须可独立运行（除声明的依赖外无外部模块导入）。
- **不自动上传数据集**：用户自己提供数据路径；工具可生成示例桩但不管理数据托管。
- **🚫 不硬编码问题列表**：对话流程由 Agent 动态驱动。
- **🚫 不硬编码脚本模板**：脚本由 Agent 根据需求直接生成。

---

## 成功指标

1. 用户从 `lazydspy chat` 到获得可运行的 `pipeline.py` 在 5 分钟内完成。
2. 生成的脚本通过 `ruff check` 和 `mypy --strict` 零错误。
3. 优化后的 prompt 在脚本执行结束时清晰呈现（打印并保存）。
4. **代码库中不存在硬编码的问题列表或脚本模板**。
5. **新增场景支持仅需调整 Prompt 或添加 Skill，无需修改核心对话逻辑**。
6. **所有数据模型使用 pydantic v2，无裸 dict 操作**。

---

## 待定问题

- [ ] Agent 是否应支持非交互（批量）模式以适配 CI 流水线？
- [ ] 初始脚本生成后如何处理多轮迭代优化？
- [ ] 可接受哪些遥测数据来改进默认推荐？
- [ ] 如何定义 Agent Skills 的边界？哪些能力应内置，哪些可扩展？
- [ ] Subagent 的使用场景和触发条件如何设计？

---

## 参考资料

- [DSPy 文档]
  - https://dspy.ai/api/
  - https://dspy.ai/tutorials/
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) – Agent 开发框架