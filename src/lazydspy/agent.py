"""Agent core module using Claude Agent SDK.

Provides the main Agent class for running multi-turn conversations.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from lazydspy.prompts import get_system_prompt_config
from lazydspy.tools import TOOL_NAMES, create_mcp_server

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions


@dataclass
class AgentConfig:
    """Configuration for the Agent."""

    model: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    )
    debug: bool = field(
        default_factory=lambda: os.environ.get("LAZYDSPY_DEBUG", "").lower() in ("1", "true", "yes")
    )
    base_url: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL"))
    workdir: Path = field(default_factory=Path.cwd)
    max_turns: int = 50

    @property
    def api_key(self) -> str | None:
        """Get API key with priority: ANTHROPIC_AUTH_TOKEN > ANTHROPIC_API_KEY."""
        return os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")


class Agent:
    """lazydspy Agent using Claude Agent SDK.

    Manages multi-turn conversations for DSPy script generation.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the Agent.

        Args:
            config: Agent configuration. If None, uses defaults from environment.
        """
        self.config = config or AgentConfig()
        self.console = Console()

    def _create_options(self) -> "ClaudeAgentOptions":
        """Create ClaudeAgentOptions for the SDK client."""
        from claude_agent_sdk import ClaudeAgentOptions

        # SDK built-in tools + MCP tools
        allowed_tools = [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            *TOOL_NAMES,
        ]

        env: dict[str, str] = {}
        if self.config.model:
            env["ANTHROPIC_MODEL"] = self.config.model
        if self.config.base_url:
            env["ANTHROPIC_BASE_URL"] = self.config.base_url
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if auth_token:
            env["ANTHROPIC_AUTH_TOKEN"] = auth_token
        elif api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        if self.config.debug:
            env["LAZYDSPY_DEBUG"] = "1"

        return ClaudeAgentOptions(
            system_prompt=get_system_prompt_config(),
            mcp_servers={"lazydspy": create_mcp_server()},
            allowed_tools=allowed_tools,
            max_turns=self.config.max_turns,
            env=env,
            cwd=str(self.config.workdir),
        )

    def _display_welcome(self) -> None:
        """Display welcome message."""
        self.console.print(
            Panel(
                "[bold green]lazydspy[/] - DSPy 优化脚本生成器\n\n"
                "我会帮你通过对话生成 DSPy prompt 优化脚本。\n"
                "输入 [bold]exit[/] 或 [bold]quit[/] 退出。",
                title="欢迎",
                border_style="blue",
            )
        )
        if self.config.debug:
            self.console.print(
                Panel(
                    "[bold]调试信息[/]\n"
                    f"model: {self.config.model}\n"
                    f"base_url: {self.config.base_url or 'default'}\n"
                    f"workdir: {self.config.workdir}\n"
                    f"max_turns: {self.config.max_turns}",
                    title="Debug",
                    border_style="yellow",
                )
            )

    def _display_response(self, text: str) -> None:
        """Display assistant response."""
        self.console.print(Markdown(text))
        self.console.print()

    async def run_async(self) -> None:
        """Run the agent conversation loop asynchronously."""
        from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, TextBlock

        self._display_welcome()
        options = self._create_options()

        async with ClaudeSDKClient(options=options) as client:
            while True:
                try:
                    user_input = self.console.input("[bold cyan]你:[/] ").strip()
                except (KeyboardInterrupt, EOFError):
                    self.console.print("\n[yellow]再见！[/]")
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q", "退出"):
                    self.console.print("[yellow]再见！[/]")
                    break

                # Send query and receive response
                await client.query(user_input)

                response_text = ""
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                response_text += block.text

                if response_text:
                    self._display_response(response_text)

    def run(self) -> None:
        """Run the agent conversation loop (synchronous wrapper)."""
        asyncio.run(self.run_async())


def run_agent(
    *,
    model: str | None = None,
    debug: bool | None = None,
    workdir: Path | None = None,
) -> None:
    """Convenience function to run the agent.

    Args:
        model: Model name override
        debug: Enable debug mode
        workdir: Working directory
    """
    defaults = AgentConfig()
    config = AgentConfig(
        model=model or defaults.model,
        debug=defaults.debug if debug is None else debug,
        base_url=defaults.base_url,
        workdir=workdir or defaults.workdir,
        max_turns=defaults.max_turns,
    )
    agent = Agent(config)
    agent.run()


__all__ = [
    "Agent",
    "AgentConfig",
    "run_agent",
]
