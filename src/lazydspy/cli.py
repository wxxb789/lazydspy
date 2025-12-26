"""Command-line interface for lazydspy.

Thin wrapper around Agent that provides CLI argument handling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from lazydspy.agent import Agent, AgentConfig

console = Console()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="lazydspy - DSPy 优化脚本生成器",
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

    # Run the agent
    agent = Agent(config)
    agent.run()


@app.callback(invoke_without_command=True)
def default_callback(
    ctx: typer.Context,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Claude model name"),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug mode"),
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
    """lazydspy - DSPy 优化脚本生成器.

    Without a subcommand, runs chat by default.
    """
    if version:
        console.print("lazydspy 0.2.0")
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
