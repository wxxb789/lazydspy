# Claude Agent SDK 参考文档索引

> 本文档索引了 Claude Agent SDK 和 DSPy 的关键文档，方便开发时查阅。

## 1. Claude Agent SDK

### 1.1 官方资源

| 资源 | 链接 |
|------|------|
| GitHub 仓库 | https://github.com/anthropics/claude-agent-sdk-python |
| PyPI | https://pypi.org/project/claude-agent-sdk/ |
| 官方文档 | https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python |

### 1.2 核心示例文件

| 文件 | 说明 | 链接 |
|------|------|------|
| `quick_start.py` | 基础用法，query() 函数 | [GitHub](https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/quick_start.py) |
| `streaming_mode.py` | ClaudeSDKClient 多轮对话 | [GitHub](https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/streaming_mode.py) |
| `mcp_calculator.py` | @tool 装饰器和 MCP 服务器 | [GitHub](https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/mcp_calculator.py) |
| `hooks.py` | Hooks 用法 (PreToolUse, PostToolUse) | [GitHub](https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/hooks.py) |
| `agents.py` | 自定义 Agent 定义 | [GitHub](https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/agents.py) |
| `system_prompt.py` | System Prompt 配置方式 | [GitHub](https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/system_prompt.py) |
| `tools_option.py` | Tools 配置选项 | [GitHub](https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/tools_option.py) |
| `plugin_example.py` | Plugin 加载示例 | [GitHub](https://github.com/anthropics/claude-agent-sdk-python/blob/main/examples/plugin_example.py) |

### 1.3 关键 API

#### ClaudeAgentOptions

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    # System Prompt 配置
    system_prompt="自定义 prompt",  # 字符串
    # 或使用 preset
    system_prompt={"type": "preset", "preset": "claude_code"},
    # 或 preset + append
    system_prompt={"type": "preset", "preset": "claude_code", "append": "额外指令"},
    
    # 工具配置
    allowed_tools=["Read", "Write", "Bash"],  # 允许的工具列表
    tools=["Read", "Glob"],  # 可用工具 (不同于 allowed_tools)
    tools=[],  # 空数组禁用所有内置工具
    
    # MCP 服务器
    mcp_servers={"name": mcp_server},
    
    # Hooks
    hooks={
        "PreToolUse": [HookMatcher(...)],
        "PostToolUse": [HookMatcher(...)],
    },
    
    # 其他
    max_turns=10,
    env={"KEY": "value"},
    cwd="/path/to/workdir",
)
```

#### ClaudeSDKClient (流式多轮对话)

```python
from claude_agent_sdk import ClaudeSDKClient, AssistantMessage, TextBlock

async with ClaudeSDKClient(options=options) as client:
    # 发送查询
    await client.query("你的问题")
    
    # 接收响应
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text)
```

#### @tool 装饰器 (MCP 工具)

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("tool_name", "工具描述", {"param1": str, "param2": int})
async def my_tool(args: dict) -> dict:
    return {
        "content": [{"type": "text", "text": "结果"}]
    }

# 创建 MCP 服务器
server = create_sdk_mcp_server(
    name="my-server",
    version="1.0.0",
    tools=[my_tool],
)

# 在 options 中使用
options = ClaudeAgentOptions(
    mcp_servers={"my": server},
    allowed_tools=["mcp__my__tool_name"],  # 注意前缀格式
)
```

#### 消息类型

```python
from claude_agent_sdk import (
    AssistantMessage,  # Claude 响应
    UserMessage,       # 用户消息
    SystemMessage,     # 系统消息
    ResultMessage,     # 结果消息 (包含 total_cost_usd)
    TextBlock,         # 文本内容块
    ToolUseBlock,      # 工具调用块
    ToolResultBlock,   # 工具结果块
)
```

### 1.4 内置工具

SDK 提供的内置工具（无需 MCP）：

| 工具名 | 说明 |
|--------|------|
| `Read` | 读取文件 |
| `Write` | 写入文件 |
| `Edit` | 编辑文件 |
| `Bash` | 执行 shell 命令 |
| `Glob` | 文件模式匹配搜索 |
| `Grep` | 内容搜索 |
| `Task` | 启动子 Agent |
| `WebFetch` | 获取网页内容 |

### 1.5 Hooks

```python
from claude_agent_sdk import HookMatcher, HookInput, HookContext, HookJSONOutput

async def my_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext
) -> HookJSONOutput:
    # PreToolUse: 可以 deny 或 allow
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",  # 或 "allow"
            "permissionDecisionReason": "原因",
        }
    }
    
    # PostToolUse: 可以添加上下文或停止执行
    return {
        "continue_": False,  # 停止执行
        "stopReason": "停止原因",
    }

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [HookMatcher(matcher="Bash", hooks=[my_hook])],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[my_hook])],  # 匹配所有
        "UserPromptSubmit": [HookMatcher(matcher=None, hooks=[...])],
    }
)
```

## 2. DSPy

### 2.1 官方资源

| 资源 | 链接 |
|------|------|
| 官网 | https://dspy.ai |
| API 文档 | https://dspy.ai/api/ |
| 教程 | https://dspy.ai/tutorials/ |
| GitHub | https://github.com/stanfordnlp/dspy |

### 2.2 核心概念

#### Signature (签名)

```python
import dspy

class QA(dspy.Signature):
    """Answer the question based on the context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

#### Module (模块)

```python
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(QA)
    
    def forward(self, question):
        context = self.retrieve(question)
        return self.generate(context=context, question=question)
```

#### Optimizer (优化器)

```python
# GEPA - 低成本进化优化
from dspy.teleprompt import GEPA
optimizer = GEPA(metric=my_metric, breadth=2, depth=2)

# MIPROv2 - 模型指令优化
from dspy.teleprompt import MIPROv2
optimizer = MIPROv2(metric=my_metric, num_candidates=8)

# 编译
optimized_module = optimizer.compile(module, trainset=train_data)
```

### 2.3 Metric 函数

```python
def my_metric(example, prediction, trace=None) -> float:
    """
    Args:
        example: 训练数据样本 (包含 gold label)
        prediction: 模型预测
        trace: 可选的执行轨迹
    
    Returns:
        0.0 到 1.0 之间的分数
    """
    return float(example.answer.lower() == prediction.answer.lower())
```

### 2.4 数据格式

```python
# JSONL 格式
{"query": "问题1", "answer": "答案1"}
{"query": "问题2", "answer": "答案2"}

# 加载数据
import json
with open("train.jsonl") as f:
    data = [json.loads(line) for line in f]

# 转为 dspy.Example
examples = [dspy.Example(**row).with_inputs("query") for row in data]
```

## 3. 项目特定参考

### 3.1 优化器预设

#### GEPA

| 模式 | breadth | depth | temperature |
|------|---------|-------|-------------|
| quick | 2 | 2 | 0.3 |
| full | 4 | 4 | 0.7 |

#### MIPROv2

| 模式 | search_size | temperature |
|------|-------------|-------------|
| quick | 8 | 0.3 |
| full | 16 | 0.6 |

### 3.2 成本估算

参考 `knowledge/cost_models.py`:
- 每次优化调用的 token 估算
- 模型价格表
- quick vs full 成本差异 (~5-10x)

### 3.3 生成脚本规范

参考 `prompts.py` 中的 SYSTEM_PROMPT_APPEND:
- PEP 723 内联元数据
- Typer CLI 参数
- Pydantic v2 DataRow 模型
- Checkpoint 支持
