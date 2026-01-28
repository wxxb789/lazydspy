"""Command-line interface for lazydspy.

Thin wrapper around Agent that provides CLI argument handling.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console

from lazydspy import __version__
from lazydspy.agent import Agent, AgentConfig

console = Console()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="lazydspy - DSPy optimization script generator",
)


@app.command(name="chat")
def chat(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Claude model name",
            envvar="ANTHROPIC_MODEL",
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode",
            envvar="LAZYDSPY_DEBUG",
        ),
    ] = False,
    workdir: Annotated[
        Path | None,
        typer.Option(
            "--workdir",
            "-w",
            help="Working directory",
        ),
    ] = None,
) -> None:
    """Start interactive conversation to generate DSPy optimization scripts.

    The Agent dynamically asks questions to gather requirements, then generates
    a ready-to-run Python script. Generated scripts are saved to generated/<session_id>/.

    Examples:
        lazydspy chat
        lazydspy chat --model claude-sonnet-4-20250514
        lazydspy chat --debug
    """
    # Validate API key
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[red]Error: API token not set[/]\n\n"
            "Please set one of these environment variables:\n"
            "  - ANTHROPIC_AUTH_TOKEN\n"
            "  - ANTHROPIC_API_KEY"
        )
        raise typer.Exit(1)

    # Build config
    config = AgentConfig(
        model=model or AgentConfig().model,
        debug=debug,
        workdir=workdir or Path.cwd(),
    )

    # Generate session ID (timestamp-based for uniqueness and readability)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run the Rich Live UI
    asyncio.run(_run_live_ui(config, session_id))


async def _run_live_ui(config: AgentConfig, session_id: str) -> None:
    """Run the Rich Live UI with persistent ClaudeSDKClient session.

    Args:
        config: Agent configuration
        session_id: Unique session identifier
    """
    # Lazy import to avoid loading UI dependencies unless chat is used
    from claude_agent_sdk import ClaudeSDKClient

    from lazydspy.tui.live_ui import LiveChatUI

    # Create UI
    ui = LiveChatUI(
        session_id=session_id,
        model=config.model,
        workdir=config.workdir,
    )

    # Create Agent with event sink
    agent = Agent(config, event_sink=ui)

    # Create persistent ClaudeSDKClient
    options = agent._create_options()

    async with ClaudeSDKClient(options=options) as client:
        # Current task for cancellation
        current_task: asyncio.Task[None] | None = None

        async def send_message(text: str) -> None:
            """Send user message to agent and stream response."""
            nonlocal current_task

            try:
                # Import stage enum
                from lazydspy.state import ConversationStage

                # Handle CONFIRM stage specially
                if agent.state.stage == ConversationStage.CONFIRM:
                    confirmed = agent._is_affirmative(text)
                    if confirmed:
                        agent._set_stage(ConversationStage.GENERATE)
                    else:
                        agent._set_stage(ConversationStage.COLLECT)

                    message = agent._build_spec_confirmation_message(
                        confirmed=confirmed,
                        feedback=None if confirmed else text,
                    )

                    # Emit GenerationStarted
                    from lazydspy.tui.events import GenerationStarted

                    agent._emit_event(GenerationStarted())

                    await client.query(message)
                    await agent._collect_response(client)

                    # Emit GenerationCompleted
                    from lazydspy.tui.events import GenerationCompleted

                    agent._emit_event(GenerationCompleted())

                    # Cast to widen type for mypy after mutations
                    if cast(ConversationStage, agent.state.stage) == ConversationStage.VALIDATE:
                        await agent._run_validation(client)

                else:
                    # Normal message flow
                    from lazydspy.tui.events import GenerationStarted

                    agent._emit_event(GenerationStarted())

                    await client.query(text)
                    await agent._collect_response(client)

                    # Emit GenerationCompleted
                    from lazydspy.tui.events import GenerationCompleted

                    agent._emit_event(GenerationCompleted())

                    # Cast to widen type for mypy after mutations
                    if cast(ConversationStage, agent.state.stage) == ConversationStage.VALIDATE:
                        await agent._run_validation(client)

            except asyncio.CancelledError:
                # Emit GenerationCancelled
                from lazydspy.tui.events import GenerationCancelled

                agent._emit_event(GenerationCancelled())
                raise

        def cancel_generation() -> None:
            """Cancel current generation."""
            if current_task and not current_task.done():
                current_task.cancel()

        # Set handlers
        ui.set_handlers(send_message=send_message, cancel_generation=cancel_generation)

        # Run UI
        await ui.run()


@app.callback(invoke_without_command=True)
def default_callback(
    ctx: typer.Context,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Claude model name", envvar="ANTHROPIC_MODEL"),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug mode", envvar="LAZYDSPY_DEBUG"),
    ] = False,
    workdir: Annotated[
        Path | None,
        typer.Option("--workdir", "-w", help="Working directory"),
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version"),
    ] = False,
) -> None:
    """lazydspy - DSPy optimization script generator.

    Without a subcommand, runs chat by default.
    """
    if version:
        console.print(f"lazydspy {__version__}")
        raise typer.Exit(0)

    if ctx.invoked_subcommand is None:
        # No subcommand, run chat by default
        chat(
            model=model,
            debug=debug,
            workdir=workdir,
        )


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
