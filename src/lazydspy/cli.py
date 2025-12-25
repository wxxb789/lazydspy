"""Command-line interface for lazydspy.

This is the completely rewritten CLI that follows the Agentic architecture.
All hardcoded question lists and script templates have been removed.
The Agent now drives the conversation dynamically.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich.console import Console

from lazydspy.agent import AgentConfig, run_agent

console = Console()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="lazydspy - Generate DSPy optimization scripts through conversation",
)


@app.command(name="chat")
def chat(
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m",
            help="Claude model name",
            envvar="ANTHROPIC_MODEL",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Custom API endpoint",
            envvar="ANTHROPIC_BASE_URL",
        ),
    ] = None,
    auth_token: Annotated[
        str | None,
        typer.Option(
            "--auth-token",
            help="API token (or set ANTHROPIC_API_KEY env var)",
            envvar="ANTHROPIC_API_KEY",
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode",
        ),
    ] = False,
) -> None:
    """Start interactive conversation to generate DSPy optimization scripts.

    The Agent dynamically asks questions to gather requirements, then generates
    a ready-to-run Python script. Generated scripts are saved to generated/<session_id>/.

    Examples:
        lazydspy chat
        lazydspy chat --model claude-opus-4.5
        lazydspy chat --debug

    """
    # Build config from CLI args and environment
    config = AgentConfig(
        model=model or os.getenv("ANTHROPIC_MODEL", "claude-opus-4.5"),
        base_url=base_url,
        auth_token=auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"),
        debug=debug,
    )

    # Validate auth token
    if not config.auth_token:
        console.print(
            "[red]Error: API token not set[/]\n\n"
            "Please provide via one of:\n"
            "  1. Environment variable: export ANTHROPIC_API_KEY=your-key\n"
            "  2. Command line argument: --auth-token your-key"
        )
        raise typer.Exit(1)

    # Run the agent
    run_agent(console=console, config=config)


@app.callback(invoke_without_command=True)
def default_callback(
    ctx: typer.Context,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Claude model name"),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Custom API endpoint"),
    ] = None,
    auth_token: Annotated[
        str | None,
        typer.Option("--auth-token", help="API token"),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug mode"),
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version"),
    ] = False,
) -> None:
    """lazydspy - Generate DSPy optimization scripts through conversation.

    Without a subcommand, runs chat by default.
    """
    if version:
        console.print("lazydspy 0.1.0")
        raise typer.Exit(0)

    if ctx.invoked_subcommand is None:
        # No subcommand, run chat by default
        chat(
            model=model,
            base_url=base_url,
            auth_token=auth_token,
            debug=debug,
        )


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
