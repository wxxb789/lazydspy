# lazydspy SDK 迁移设计文档

> 版本: 0.2.0  
> 日期: 2025-12-26  
> 状态: 实施中

## 1. 概述

### 1.1 背景

lazydspy 原使用 `anthropic.AsyncAnthropic` 直接调用 Claude API，手动管理工具调用循环。这导致了：
- 工具参数验证不完善（`KeyError: 'content'` 问题）
- 代码冗余（自定义 file_ops 等工具）
- 维护负担重

### 1.2 目标

迁移到 **Claude Agent SDK**，利用其内置能力：
- 使用 SDK 内置工具（Read, Write, Bash, Glob, Grep）
- 使用 `@tool` 装饰器定义业务工具
- 使用 `ClaudeSDKClient` 管理多轮对话
- 代码量从 ~1850 行减少到 ~300 行

## 2. 新架构

### 2.1 目录结构

```
src/lazydspy/
├── __init__.py       # 包导出
├── __main__.py       # 入口点
├── cli.py            # Typer CLI (~50行)
├── agent.py          # Agent 核心 (~100行)
├── tools.py          # MCP 工具 (~80行)
├── prompts.py        # System Prompt (~60行)
└── knowledge/        # 领域知识 (保持不变)
    ├── __init__.py
    ├── cost_models.py
    └── optimizers.py
```

### 2.2 核心模块

#### `tools.py` - MCP 工具定义

使用 `@tool` 装饰器定义业务工具，通过 `create_sdk_mcp_server()` 创建 MCP 服务器。

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("estimate_cost", "估算 DSPy 优化成本", {...})
async def estimate_cost(args: dict) -> dict:
    ...

def create_mcp_server():
    return create_sdk_mcp_server(
        name="lazydspy",
        version="0.1.0", 
        tools=[estimate_cost, list_optimizers, get_defaults],
    )
```

#### `prompts.py` - System Prompt

使用 `preset` + `append` 模式扩展 claude_code 预设。

```python
def get_system_prompt_config() -> dict:
    return {
        "type": "preset",
        "preset": "claude_code",
        "append": SYSTEM_PROMPT_APPEND,
    }
```

#### `agent.py` - Agent 核心

使用 `ClaudeSDKClient` 实现多轮对话。

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

class Agent:
    async def run(self) -> None:
        options = ClaudeAgentOptions(
            system_prompt=get_system_prompt_config(),
            mcp_servers={"lazydspy": create_mcp_server()},
            allowed_tools=[...],
        )
        
        async with ClaudeSDKClient(options=options) as client:
            while True:
                user_input = input("你: ")
                await client.query(user_input)
                async for msg in client.receive_response():
                    self._display(msg)
```

## 3. 删除清单

### 3.1 删除的目录

| 目录 | 文件数 | 原因 |
|------|--------|------|
| `agent/` | 5 | 合并到 agent.py |
| `tools/` | 5 | 合并到 tools.py，file_ops 使用 SDK 内置 |
| `models/` | 3 | 迁移后不需要 |

### 3.2 删除的文件

| 文件 | 原因 |
|------|------|
| `schemas.py` | 不再需要 |
| `src/core.py` | Legacy stub |
| `src/dspy.py` | Legacy stub |
| `src/metrics.py` | Legacy stub |
| `src/pydantic.py` | Legacy stub |

## 4. 关键设计决策

### 4.1 System Prompt

**决策**: 使用 `preset: claude_code` + `append`

**理由**:
- 继承 claude_code 的文件操作能力
- 只需添加 lazydspy 特定指令
- 无需重复定义工具使用说明

### 4.2 文件操作

**决策**: 使用 SDK 内置工具 (Read, Write, Edit, Bash)

**理由**:
- SDK 工具经过充分测试
- 减少代码维护负担
- 自动处理权限和错误

### 4.3 Session 管理

**决策**: 删除 session.py，使用 SDK 内部管理

**理由**:
- `ClaudeSDKClient` 自动管理会话状态
- 减少重复代码

### 4.4 finish_session 工具

**决策**: 删除，不需要显式完成信号

**理由**:
- 多轮对话模式下用户随时可输入 `exit` 退出
- Agent 完成任务后自然回到等待输入状态

### 4.5 环境变量

**决策**: 
- 优先级: `ANTHROPIC_AUTH_TOKEN` > `ANTHROPIC_API_KEY`
- 支持 `ANTHROPIC_BASE_URL` 自定义端点
- 支持 `ANTHROPIC_MODEL` 指定模型

## 5. 依赖变化

### 5.1 新依赖

```toml
dependencies = [
    "claude-agent-sdk>=0.1.17",
    "rich",
    "typer>=0.12",
]
```

### 5.2 移除的依赖

- `anthropic` - SDK 已包含
- `dspy` - 生成脚本需要，lazydspy 本身不需要
- `openai` - 同上
- `pydantic` - SDK 已包含
- `python-dotenv` - 不需要

## 6. 测试策略

### 6.1 保留的测试

- `knowledge/` 相关测试 (cost_models, optimizers)

### 6.2 新增的测试

- `tools.py` 工具函数测试
- `agent.py` 配置测试
- SDK stub 用于隔离测试

### 6.3 删除的测试

- 所有涉及 `agent/`, `tools/`, `models/` 旧模块的测试

## 7. 迁移步骤

### Phase 1: 文档准备
- [x] 创建 migration-design.md
- [ ] 创建 sdk-reference.md

### Phase 2: 新架构实现
- [ ] 创建 tools.py
- [ ] 创建 prompts.py
- [ ] 创建 agent.py
- [ ] 重写 cli.py
- [ ] 更新 __init__.py
- [ ] 更新 __main__.py

### Phase 3: 清理旧代码
- [ ] 删除 agent/ 目录
- [ ] 删除 tools/ 目录
- [ ] 删除 models/ 目录
- [ ] 删除 schemas.py
- [ ] 删除 legacy 文件

### Phase 4: 配置更新
- [ ] 更新 pyproject.toml
- [ ] 更新 .env.example

### Phase 5: 测试
- [ ] 更新测试文件
- [ ] 运行测试验证

### Phase 6: 文档完善
- [ ] 更新 CLAUDE.md
- [ ] 更新 README.md
