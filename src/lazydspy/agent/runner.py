"""Agent runner - the core agent driver."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from lazydspy.tools import SessionComplete, get_all_tool_schemas, get_tool_handler

from .config import AgentConfig
from .prompts import SYSTEM_PROMPT
from .session import APIMessage, ConversationSession


class AgentRunner:
    """Main agent driver that orchestrates conversations and tool execution."""

    def __init__(
        self,
        console: Console,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the agent runner.

        Args:
            console: Rich console for output
            config: Agent configuration (uses env vars if not provided)
        """
        self.console = console
        self.config = config or AgentConfig.from_env()
        self.session = ConversationSession()
        self._client: Any = None
        # Session completion state
        self._session_complete = False
        self._completion_summary = ""
        self._completion_next_steps: list[str] = []

    def _get_client(self) -> Any:
        """Get or create the Anthropic async client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.config.auth_token,
                    base_url=self.config.base_url,
                )
            except ImportError as err:
                raise RuntimeError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                ) from err
        return self._client

    async def _call_claude(self, messages: list[APIMessage]) -> Any:
        """Call Claude API with messages and tools.

        Args:
            messages: Conversation messages

        Returns:
            Claude API response
        """
        client = self._get_client()

        # Get tool schemas
        tools = get_all_tool_schemas()

        # Make async API call
        response = await client.messages.create(
            model=self.config.model,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )

        return response

    async def _execute_tool(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool input arguments

        Returns:
            Tuple of (tool result, session_complete flag)
        """
        handler = get_tool_handler(tool_name)
        if handler is None:
            return {
                "content": [{
                    "type": "text",
                    "text": f"未知工具: {tool_name}",
                }]
            }, False

        try:
            result = await handler(tool_input)
            self.session.add_tool_result(tool_name, result)
            return result, False
        except SessionComplete as e:
            # Session completion signal
            self._session_complete = True
            self._completion_summary = e.summary
            self._completion_next_steps = e.next_steps
            return {
                "content": [{
                    "type": "text",
                    "text": f"会话完成: {e.summary}",
                }]
            }, True
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"工具执行失败: {e}",
                }]
            }, False

    async def _process_response(self, response: Any) -> tuple[str | None, bool]:
        """Process a Claude response, executing tools if needed.

        Args:
            response: Claude API response

        Returns:
            Tuple of (assistant message text, whether to continue conversation)
        """
        assistant_text: list[str] = []
        tool_use_blocks: list[dict[str, Any]] = []

        # Extract text and tool use blocks
        for block in response.content:
            if block.type == "text":
                assistant_text.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Display assistant text
        if assistant_text:
            combined_text = "\n".join(assistant_text)
            self.console.print(Panel(
                Markdown(combined_text),
                title="🤖 Agent",
                border_style="blue",
            ))
            self.session.add_assistant_message(combined_text)

        # Execute tools if any
        if tool_use_blocks:
            tool_results = []
            session_complete = False

            for tool_block in tool_use_blocks:
                tool_name = tool_block["name"]
                tool_input = tool_block["input"]

                self.console.print(f"[cyan]⚡ 执行工具: {tool_name}[/]")
                if self.config.debug:
                    input_str = json.dumps(tool_input, ensure_ascii=False)
                    self.console.print(f"[dim]   输入: {input_str}[/]")

                result, is_complete = await self._execute_tool(tool_name, tool_input)

                if is_complete:
                    session_complete = True

                # Extract text from result
                result_text = ""
                if "content" in result and result["content"]:
                    for item in result["content"]:
                        if item.get("type") == "text":
                            result_text = item.get("text", "")
                            break

                if result_text:
                    # Show abbreviated result
                    if len(result_text) > 200:
                        display_text = result_text[:200] + "..."
                    else:
                        display_text = result_text
                    self.console.print(f"[green]   ✓ {display_text}[/]")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block["id"],
                    "content": result_text,
                })

            # If session is complete, return immediately
            if session_complete:
                return "\n".join(assistant_text) if assistant_text else None, False

            # Continue conversation with tool results
            messages = self.session.get_messages()

            # Add the assistant's response including tool use
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Add tool results
            messages.append({
                "role": "user",
                "content": tool_results,
            })

            # Get next response
            next_response = await self._call_claude(messages)
            return await self._process_response(next_response)

        # Check if conversation should continue
        stop_reason = response.stop_reason
        should_continue = stop_reason != "end_turn"

        return "\n".join(assistant_text) if assistant_text else None, should_continue

    async def run_conversation(self) -> None:
        """Run the interactive conversation loop."""
        # Reset session completion state
        self._session_complete = False
        self._completion_summary = ""
        self._completion_next_steps = []

        self.console.print(Panel(
            "[bold]欢迎使用 lazydspy![/bold]\n\n"
            "我将帮助你生成一个 DSPy 优化脚本。\n"
            "请描述你的需求，我会通过对话收集必要信息。\n\n"
            "[dim]输入 'exit' 或 'quit' 退出，输入 'help' 获取帮助[/]",
            title="🚀 lazydspy Agent",
            border_style="green",
        ))

        turn_count = 0

        while turn_count < self.config.max_turns:
            try:
                # Get user input
                self.console.print()
                user_input = input("你: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in {"exit", "quit", "q"}:
                    self.console.print("[yellow]👋 再见！[/]")
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                # Add user message
                self.session.add_user_message(user_input)

                # Call Claude
                messages = self.session.get_messages()
                response = await self._call_claude(messages)

                # Process response
                _, _ = await self._process_response(response)

                # Check if session completed after processing
                if self._session_complete:
                    self._show_completion_summary()
                    self._session_complete = False  # Allow new tasks

                turn_count += 1

            except KeyboardInterrupt:
                self.console.print("\n[yellow]👋 已中断，再见！[/]")
                break
            except Exception as e:
                self.console.print(f"[red]❌ 错误: {e}[/]")
                if self.config.debug:
                    import traceback
                    self.console.print(f"[dim]{traceback.format_exc()}[/]")

        if turn_count >= self.config.max_turns:
            self.console.print("[yellow]⚠️ 达到最大对话轮数限制[/]")

    def _show_completion_summary(self) -> None:
        """Display completion summary to user."""
        lines = ["[bold green]任务完成！[/]", "", self._completion_summary]

        if self._completion_next_steps:
            lines.extend(["", "[bold]下一步建议：[/]"])
            for step in self._completion_next_steps:
                lines.append(f"  • {step}")

        lines.extend(["", "[dim]您可以继续提出新需求，或输入 'exit' 退出。[/]"])

        self.console.print(Panel("\n".join(lines), title="✅ 任务完成", border_style="green"))

    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
## 可用命令

- `exit` / `quit` / `q` - 退出对话
- `help` - 显示此帮助信息

## 使用说明

1. 描述你想要优化的任务场景
2. 回答 Agent 的问题以提供必要信息
3. 确认配置后 Agent 会生成脚本

## 示例对话开始语

- "我想优化一个文本摘要任务"
- "帮我生成一个检索问答的优化脚本"
- "我需要优化一个分类模型的 prompt"
        """
        self.console.print(Panel(
            Markdown(help_text),
            title="帮助",
            border_style="cyan",
        ))


def run_agent(
    console: Console | None = None,
    config: AgentConfig | None = None,
) -> None:
    """Run the agent (synchronous wrapper).

    Args:
        console: Rich console (creates one if not provided)
        config: Agent config (uses env vars if not provided)
    """
    if console is None:
        console = Console()

    runner = AgentRunner(console=console, config=config)
    asyncio.run(runner.run_conversation())


__all__ = ["AgentRunner", "run_agent"]
