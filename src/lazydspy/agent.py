"""Agent core module using Claude Agent SDK.

Provides the main Agent class for running multi-turn conversations.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from lazydspy.prompts import get_system_prompt_config
from lazydspy.state import AgentState, ConversationStage
from lazydspy.tools import TOOL_NAMES, bind_state, create_mcp_server

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient


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
        self.state = AgentState()
        bind_state(self.state)

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

    async def _collect_response(self, client: "ClaudeSDKClient") -> str:
        """Collect assistant response text from the SDK client."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        response_text = ""
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
        return response_text

    def _is_affirmative(self, text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in {
            "y",
            "yes",
            "ok",
            "okay",
            "是",
            "确认",
            "好的",
            "可以",
            "没问题",
            "行",
        }

    def _build_spec_confirmation_message(self, confirmed: bool, feedback: str | None = None) -> str:
        spec = self.state.spec
        if not spec:
            self.state.stage = ConversationStage.COLLECT
            return "当前没有可确认的规范，请继续收集需求并调用 mcp__lazydspy__submit_spec。"

        spec_json = json.dumps(spec.model_dump(), ensure_ascii=False, indent=2)
        if confirmed:
            return (
                "用户已确认以下规范，请开始生成脚本。\n"
                "生成完成后，请调用 mcp__lazydspy__mark_generation_complete "
                "并提交生成文件列表。\n\n"
                f"{spec_json}"
            )

        feedback_text = feedback.strip() if feedback else "用户未确认"
        return (
            "用户未确认规范，请根据反馈继续提问并修订。\n"
            "修订完成后再次调用 mcp__lazydspy__submit_spec。\n\n"
            f"用户反馈: {feedback_text}\n\n"
            f"{spec_json}"
        )

    def _validate_generated_files(self) -> list[str]:
        errors: list[str] = []
        spec = self.state.spec
        if not spec:
            errors.append("未找到已确认的规范")
            return errors

        if not self.state.generated_files:
            errors.append("未收到生成文件列表")
            return errors

        py_files = [path for path in self.state.generated_files if path.suffix == ".py"]
        target_files = [path for path in py_files if path.name == "pipeline.py"] or py_files
        if not target_files:
            errors.append("未找到需要校验的 Python 脚本")
            return errors

        for path in target_files:
            if not path.exists():
                errors.append(f"文件不存在: {path}")
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            if "# ///" not in text or "dependencies" not in text:
                errors.append(f"{path}: 缺少 PEP 723 元数据块")

            for dep in ("dspy", "pydantic", "typer", "rich"):
                if dep not in text:
                    errors.append(f"{path}: 依赖未声明或未使用: {dep}")

            if "typer" not in text:
                errors.append(f"{path}: 缺少 Typer CLI")
            if "--mode" not in text:
                errors.append(f"{path}: 缺少 --mode 参数")
            if "--data" not in text and "--data-path" not in text:
                errors.append(f"{path}: 缺少 --data 参数")
            if "--checkpoint-dir" not in text and "checkpoint" not in text:
                errors.append(f"{path}: 缺少 checkpoint 参数")
            if "--resume" not in text and "resume" not in text:
                errors.append(f"{path}: 缺少 resume 参数")
            if "jsonl" not in text.lower():
                errors.append(f"{path}: 缺少 JSONL 数据处理逻辑")
            if "DataRow" not in text or "BaseModel" not in text:
                errors.append(f"{path}: 缺少 Pydantic DataRow 定义")

        return errors

    async def _run_validation(self, client: "ClaudeSDKClient") -> None:
        errors = self._validate_generated_files()
        if errors:
            self.state.last_validation_errors = errors
            self.state.stage = ConversationStage.GENERATE
            error_text = "\n".join(f"- {item}" for item in errors)
            message = (
                "生成脚本未通过校验，请修复以下问题后重新生成并调用 "
                "mcp__lazydspy__mark_generation_complete：\n"
                f"{error_text}"
            )
            await client.query(message)
            response_text = await self._collect_response(client)
            if response_text:
                self._display_response(response_text)
        else:
            self.state.last_validation_errors = []
            self.state.stage = ConversationStage.DONE
            self.console.print("[green]脚本校验通过，可以运行生成脚本。[/]")

    async def run_async(self) -> None:
        """Run the agent conversation loop asynchronously."""
        from claude_agent_sdk import ClaudeSDKClient

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

                if self.state.stage == ConversationStage.CONFIRM:
                    confirmed = self._is_affirmative(user_input)
                    if confirmed:
                        self.state.stage = ConversationStage.GENERATE
                    else:
                        self.state.stage = ConversationStage.COLLECT

                    message = self._build_spec_confirmation_message(
                        confirmed=confirmed,
                        feedback=None if confirmed else user_input,
                    )
                    await client.query(message)
                    response_text = await self._collect_response(client)
                    if response_text:
                        self._display_response(response_text)
                    if self.state.stage == ConversationStage.VALIDATE:
                        await self._run_validation(client)
                    continue

                # Send query and receive response
                await client.query(user_input)

                response_text = await self._collect_response(client)
                if response_text:
                    self._display_response(response_text)

                if self.state.stage == ConversationStage.VALIDATE:
                    await self._run_validation(client)

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
