"""Live UI components for streaming agent responses.

This module provides:
- TurnState: Immutable state for a single agent turn
- reduce_event(): Pure reducer function for state updates
- Rich renderables: ToolsBulletList, AssistantBlock, StatusLine
- LiveChatUI: Main chat loop with Rich Live rendering and prompt_toolkit input
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Coroutine, Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import has_completions
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from lazydspy.state import ConversationStage
from lazydspy.tui.completions import (
    CombinedCompleter,
    FileReferenceCompleter,
    SlashCommandCompleter,
)
from lazydspy.tui.events import (
    AssistantTextDelta,
    ErrorEvent,
    Event,
    GenerationCancelled,
    GenerationCompleted,
    GenerationStarted,
    StageChanged,
    ToolCallFinished,
    ToolCallIssued,
    UserMessageSubmitted,
)
from lazydspy.tui.session import export_session_json
from lazydspy.tui.viewmodel import ViewModel


@dataclass
class ToolCallState:
    """State for a single tool call."""

    call_id: str
    name: str
    status: Literal["running", "ok", "error"] = "running"
    input_text: str = ""
    result_summary: str | None = None
    started_at: float = field(default_factory=time.time)


@dataclass
class TurnState:
    """Immutable state for a single agent turn."""

    assistant_buffer: str = ""
    tool_calls: dict[str, ToolCallState] = field(default_factory=dict)
    tool_order: list[str] = field(default_factory=list)  # preserve first-seen order
    generating: bool = False
    stage: ConversationStage = ConversationStage.COLLECT
    last_error: str | None = None
    completed: bool = False


def reduce_event(state: TurnState, event: Event) -> TurnState:
    """Pure reducer: given state and event, return new state.

    Args:
        state: Current turn state
        event: Event to process

    Returns:
        New turn state (does not mutate input)
    """
    if isinstance(event, AssistantTextDelta):
        return replace(state, assistant_buffer=state.assistant_buffer + event.text)

    if isinstance(event, ToolCallIssued):
        # Determine input text
        if event.input_text:
            input_text = event.input_text
        elif event.input_delta:
            existing = state.tool_calls.get(event.call_id)
            input_text = (existing.input_text if existing else "") + event.input_delta
        else:
            existing = state.tool_calls.get(event.call_id)
            input_text = existing.input_text if existing else ""

        # Create or update tool call
        tool = state.tool_calls.get(event.call_id)
        if tool is None:
            tool = ToolCallState(
                call_id=event.call_id,
                name=event.tool_name,
                input_text=input_text,
            )
            new_calls = {**state.tool_calls, event.call_id: tool}
            new_order = [*state.tool_order, event.call_id]
            return replace(state, tool_calls=new_calls, tool_order=new_order)
        else:
            updated_tool = replace(tool, input_text=input_text)
            new_calls = {**state.tool_calls, event.call_id: updated_tool}
            return replace(state, tool_calls=new_calls)

    if isinstance(event, ToolCallFinished):
        tool = state.tool_calls.get(event.call_id)
        if tool is None:
            # Tool not found, create placeholder
            tool = ToolCallState(call_id=event.call_id, name="unknown")
            new_calls = {**state.tool_calls, event.call_id: tool}
            new_order = [*state.tool_order, event.call_id]
            state = replace(state, tool_calls=new_calls, tool_order=new_order)
            tool = new_calls[event.call_id]

        status: Literal["ok", "error"] = "error" if event.is_error else "ok"
        updated_tool = replace(tool, status=status, result_summary=event.result_summary)
        new_calls = {**state.tool_calls, event.call_id: updated_tool}
        return replace(state, tool_calls=new_calls)

    if isinstance(event, GenerationStarted):
        return replace(state, generating=True)

    if isinstance(event, GenerationCompleted):
        return replace(state, generating=False, completed=True)

    if isinstance(event, GenerationCancelled):
        return replace(state, generating=False, completed=True)

    if isinstance(event, StageChanged):
        return replace(state, stage=event.stage)

    if isinstance(event, ErrorEvent):
        return replace(state, last_error=event.message)

    # Unknown event type, return unchanged
    return state


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _safe_glyph(glyph: str, fallback: str, console: Console) -> str:
    """Return glyph if terminal supports it, otherwise fallback.

    Args:
        glyph: Preferred Unicode glyph (e.g., "✓")
        fallback: ASCII fallback (e.g., "OK")
        console: Rich Console for encoding detection

    Returns:
        Safe glyph string for current terminal
    """
    # Check if terminal is dumb or has weak encoding
    if console.is_dumb_terminal:
        return fallback

    # Check encoding support (Windows cmd.exe often uses cp437/cp1252)
    encoding = getattr(sys.stdout, "encoding", "utf-8").lower()
    if encoding in {"ascii", "cp437", "cp1252", "latin-1"}:
        return fallback

    return glyph


def _safe_spinner(console: Console) -> str:
    """Return spinner type safe for current terminal encoding.

    Args:
        console: Rich Console for encoding detection

    Returns:
        Spinner type name compatible with terminal encoding
    """
    # Check if terminal is dumb or has weak encoding
    if console.is_dumb_terminal:
        return "line"  # ASCII-safe spinner (uses -, \, |, /)

    # Check encoding support (Windows cmd.exe often uses cp437/cp1252)
    encoding = getattr(sys.stdout, "encoding", "utf-8").lower()
    if encoding in {"ascii", "cp437", "cp1252", "latin-1"}:
        return "line"  # ASCII-safe spinner

    # UTF-8 terminals can use fancy spinners
    return "dots"  # Unicode spinner (uses Braille patterns)


def _render_compact_markdown(text: str, console: Console) -> Group:
    """Render markdown with compact code blocks (less padding, no heavy backgrounds).

    Non-code text is rendered with Rich Markdown for proper formatting (bold, lists, inline code).
    Code blocks get minimal styling (no background box, no padding).
    """
    parts: list[Markdown | Syntax] = []

    # Simple regex to extract code blocks
    code_block_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    last_end = 0

    for match in code_block_pattern.finditer(text):
        # Add text before code block as Markdown
        before = text[last_end : match.start()]
        if before.strip():
            parts.append(Markdown(before))

        # Add code block with minimal styling
        lang = match.group(1) or "text"
        code = match.group(2).rstrip()
        parts.append(
            Syntax(
                code,
                lang,
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
                background_color=None,  # No background box
                padding=0,
            )
        )
        last_end = match.end()

    # Add remaining text as Markdown
    remaining = text[last_end:]
    if remaining.strip():
        parts.append(Markdown(remaining))

    return Group(*parts) if parts else Group(Markdown(text))


class ToolsBulletList:
    """Render tool calls as kimi-style bullet list with encoding-safe glyphs."""

    def __init__(self, state: TurnState):
        self.state = state

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if not self.state.tool_order:
            return

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=3)  # bullet/spinner
        table.add_column()  # tool info

        for call_id in self.state.tool_order:
            tool = self.state.tool_calls.get(call_id)
            if not tool:
                continue

            # Bullet based on status with encoding fallback
            bullet: Spinner | Text
            if tool.status == "running":
                spinner_type = _safe_spinner(console)
                bullet = Spinner(spinner_type, style="cyan")
            elif tool.status == "ok":
                glyph = _safe_glyph("✓", "OK", console)
                bullet = Text(glyph, style="bold green")
            else:  # error
                glyph = _safe_glyph("✗", "X", console)
                bullet = Text(glyph, style="bold red")

            # Tool info line
            elapsed = time.time() - tool.started_at
            info = Text()
            info.append(tool.name, style="bold cyan")

            # Show elapsed time for completed calls
            if tool.status != "running":
                info.append(f" {elapsed:.2f}s", style="dim yellow")

            # Show truncated input
            if tool.input_text:
                truncated = _truncate(tool.input_text, 50)
                info.append(f" · {truncated}", style="dim")

            # Show result summary for completed calls
            if tool.result_summary and tool.status != "running":
                truncated = _truncate(tool.result_summary, 60)
                info.append(f" → {truncated}", style="dim italic")

            table.add_row(bullet, info)

        yield table


class AssistantBlock:
    """Render assistant response. Text during streaming, compact Markdown when done."""

    def __init__(self, state: TurnState, console: Console):
        self.state = state
        self.console = console

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if not self.state.assistant_buffer:
            return

        if self.state.generating:
            # During streaming: plain text (faster, no flicker)
            yield Text(self.state.assistant_buffer, style="white")
        else:
            # After completion: render as compact Markdown
            yield _render_compact_markdown(self.state.assistant_buffer, console)


class StatusLine:
    """Render status bar with stage, model, workdir, turn, and generation spinner."""

    def __init__(self, state: TurnState, model: str, workdir: str, turn: int):
        self.state = state
        self.model = model
        self.workdir = workdir
        self.turn = turn

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Build a single-line status using Table.grid for proper layout
        table = Table.grid(expand=True, padding=0)
        table.add_column(justify="left")  # Left: stage + status
        table.add_column(justify="right")  # Right: metadata + shortcuts

        # Left side: Stage + Generation status
        left_parts: list[Spinner | Text] = []

        # Stage text
        stage_text = Text()
        stage_text.append(f"Stage: {self.state.stage.value}", style="cyan")
        stage_text.append(" | ", style="dim")
        left_parts.append(stage_text)

        if self.state.generating:
            # Determine generation phase
            if self.state.tool_order:
                phase = "thinking"
            else:
                phase = "composing"

            # Add spinner (encoding-safe)
            spinner_type = _safe_spinner(console)
            left_parts.append(Spinner(spinner_type, style="yellow"))

            # Add phase text
            phase_text = Text(f" {phase}", style="yellow")
            left_parts.append(phase_text)
        else:
            idle_text = Text("idle", style="green")
            left_parts.append(idle_text)

        # Combine left parts into a single renderable
        # Use Table.grid to layout spinner + text horizontally
        left_table = Table.grid(padding=0)
        left_table.add_column()  # stage text
        left_table.add_column()  # spinner (if generating)
        left_table.add_column()  # phase text (if generating)

        if self.state.generating:
            left_table.add_row(left_parts[0], left_parts[1], left_parts[2])
        else:
            left_table.add_row(left_parts[0], left_parts[1], Text(""))

        # Right side: Model + Workdir + Turn + Shortcuts
        right = Text()
        right.append(f"Model: {self.model}", style="dim")
        right.append(" | ", style="dim")
        right.append(f"Workdir: {Path(self.workdir).name}", style="dim")
        right.append(" | ", style="dim")
        right.append(f"Turn: {self.turn}", style="dim")
        right.append(" | ", style="dim")
        right.append("Enter: send | Ctrl+J: newline | Esc: cancel", style="dim italic")

        table.add_row(left_table, right)
        yield table


class LiveChatUI:
    """Rich Live chat UI with prompt_toolkit input."""

    def __init__(
        self,
        session_id: str,
        model: str,
        workdir: Path,
    ) -> None:
        self.session_id = session_id
        self.model = model
        self.workdir = workdir
        self.console = Console()

        # Event queue for async communication
        self.event_queue: asyncio.Queue[Event | None] = asyncio.Queue()

        # State
        self.turn_state = TurnState()
        self.viewmodel = ViewModel(session_id=session_id, model=model, workdir=workdir)
        self.turn_count = 0

        # Cancellation guard / live state
        self._cancelling = False
        self._in_live = False

        # Handlers (set by CLI)
        self._send_message: Callable[[str], Coroutine[Any, Any, None]] | None = None
        self._cancel_generation: Callable[[], None] | None = None
        self._current_task: asyncio.Task[None] | None = None

        # Weak-terminal detection
        self._is_interactive = (
            sys.stdin.isatty()
            and sys.stdout.isatty()
            and self.console.is_terminal
            and not self.console.is_dumb_terminal
        )

        # Input session with completers and key bindings (only if interactive)
        if self._is_interactive:
            completer = CombinedCompleter(
                [
                    SlashCommandCompleter(),
                    FileReferenceCompleter(workspace_root=workdir),
                ]
            )

            # Build key bindings
            kb = KeyBindings()

            @kb.add("enter", filter=has_completions)
            def _accept_completion(event: KeyPressEvent) -> None:
                """Accept completion when Enter is pressed and completions are shown."""
                buff = event.current_buffer
                if buff.complete_state and buff.complete_state.completions:
                    # Get current completion or first one
                    completion = buff.complete_state.current_completion
                    if not completion:
                        completion = buff.complete_state.completions[0]
                    buff.apply_completion(completion)

            @kb.add("enter", filter=~has_completions)
            def _submit_message(event: KeyPressEvent) -> None:
                """Submit message when Enter is pressed without completions."""
                event.current_buffer.validate_and_handle()

            @kb.add("escape", "enter", eager=True)
            @kb.add("c-j", eager=True)
            def _insert_newline(event: KeyPressEvent) -> None:
                """Insert newline when Alt+Enter or Ctrl+J is pressed."""
                event.current_buffer.insert_text("\n")

            self.prompt_session: PromptSession[str] | None = PromptSession(
                history=InMemoryHistory(),
                completer=completer,
                multiline=True,  # Enable multiline, Enter behavior controlled by key bindings
                key_bindings=kb,
            )
        else:
            self.prompt_session = None

        # Session directory for exports
        self.session_dir = workdir / "generated" / session_id

    def emit_event(self, event: Event) -> None:
        """Add event to queue (called by Agent)."""
        self.event_queue.put_nowait(event)

    def on_event(self, event: Event) -> None:
        """EventSink protocol implementation - add event to queue."""
        self.emit_event(event)

    def set_handlers(
        self,
        send_message: Callable[[str], Coroutine[Any, Any, None]],
        cancel_generation: Callable[[], None],
    ) -> None:
        """Set message handlers (called by CLI)."""
        self._send_message = send_message
        self._cancel_generation = cancel_generation

    async def run(self) -> None:
        """Main chat loop."""
        # Welcome message
        welcome_text = (
            "lazydspy - DSPy 优化脚本生成器\n"
            "Type /help for commands. "
            "Press Esc or Ctrl+C to cancel generation."
        )

        if self._is_interactive:
            self.console.print(
                Panel(
                    f"[bold]{welcome_text}[/bold]",
                    title="Welcome",
                    border_style="blue",
                )
            )
        else:
            # Fallback mode: plain text
            print("=" * 60)
            print(welcome_text)
            print("=" * 60)
            print("Note: Running in fallback mode (non-interactive terminal)")
            print()

        while True:
            try:
                # Get user input
                if self._is_interactive and self.prompt_session:
                    # Interactive mode with prompt_toolkit
                    with patch_stdout(raw=True):
                        user_input = await self.prompt_session.prompt_async(">>> ")
                else:
                    # Fallback mode: simple input
                    print(">>> ", end="", flush=True)
                    user_input = await asyncio.get_event_loop().run_in_executor(None, input, "")

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    should_exit = self._handle_slash_command(user_input)
                    if should_exit:
                        break
                    continue

                # Regular message - send to agent
                await self._process_user_message(user_input)

            except KeyboardInterrupt:
                # Ctrl+C - cancel or exit
                if self._current_task and not self._current_task.done():
                    self._cancel_current()
                else:
                    break
            except EOFError:
                # Ctrl+D - exit
                break

        if self._is_interactive:
            self.console.print("[dim]Goodbye![/dim]")
        else:
            print("Goodbye!")

    async def _process_user_message(self, text: str) -> None:
        """Process user message and stream response."""
        if self._send_message is None:
            if self._is_interactive:
                self.console.print("[red]Error: No message handler configured[/red]")
            else:
                print("Error: No message handler configured")
            return

        # Record user message
        self.viewmodel.on_event(UserMessageSubmitted(text))
        self.turn_count += 1

        # Print user message
        if self._is_interactive:
            self.console.print(f"\n[bold cyan]You:[/bold cyan] {text}")
        else:
            print(f"\nYou: {text}")

        # Reset turn state for new response
        self.turn_state = TurnState(stage=self.turn_state.stage)

        # Start agent task
        self._current_task = asyncio.create_task(self._send_message(text))

        # Stream response with Live
        await self._stream_response()

    async def _stream_response(self) -> None:
        """Stream agent response using Rich Live.

        Supports Esc-cancel during streaming in interactive mode.
        Use Ctrl+C to cancel generation on all platforms.
        """

        def build_renderable() -> Group:
            """Build composite renderable from turn state."""
            parts: list[ToolsBulletList | AssistantBlock | StatusLine] = []

            # Tool calls (if any)
            if self.turn_state.tool_order:
                parts.append(ToolsBulletList(self.turn_state))

            # Assistant text
            if self.turn_state.assistant_buffer:
                parts.append(AssistantBlock(self.turn_state, self.console))

            # Status line
            parts.append(
                StatusLine(
                    self.turn_state,
                    self.model,
                    str(self.workdir),
                    self.turn_count,
                )
            )

            return Group(*parts)

        try:
            if self._is_interactive:
                # Interactive mode: use Rich Live with Esc-cancel support
                cancel_event = asyncio.Event()
                input_obj = None
                input_context = None

                try:
                    # Create input for Esc detection
                    input_obj = create_input()
                    input_context = input_obj.raw_mode()
                    input_context.__enter__()

                    # Get event loop for thread-safe callback
                    loop = asyncio.get_event_loop()

                    def on_input_ready() -> None:
                        """Callback when input is ready (called by prompt_toolkit)."""
                        if input_obj:
                            keys = list(input_obj.read_keys())
                            for key in keys:
                                if key.key == Keys.Escape:
                                    # Thread-safe event set
                                    loop.call_soon_threadsafe(cancel_event.set)
                                    return

                    # Attach input callback
                    input_obj.attach(on_input_ready)

                    # Throttling state for Live updates
                    last_render_at = 0.0
                    min_interval = 1 / 25  # 25 Hz max refresh rate

                    self._in_live = True
                    cancelled = False
                    with Live(
                        build_renderable(),
                        console=self.console,
                        refresh_per_second=10,  # Let Live handle spinner animation
                        transient=False,
                    ) as live:
                        while not self.turn_state.completed:
                            # Check for Esc cancel
                            if cancel_event.is_set():
                                cancelled = True
                                self._cancel_current(quiet=True)
                                self.turn_state = reduce_event(
                                    self.turn_state,
                                    GenerationCancelled(),
                                )
                                live.update(build_renderable())
                                break

                            try:
                                # Wait for event with timeout
                                event = await asyncio.wait_for(
                                    self.event_queue.get(),
                                    timeout=0.1,
                                )
                                if event is None:
                                    break

                                # Update state
                                self.turn_state = reduce_event(self.turn_state, event)
                                self.viewmodel.on_event(event)

                                # Determine if this event requires immediate redraw
                                force_redraw = isinstance(
                                    event,
                                    (
                                        ToolCallIssued,
                                        ToolCallFinished,
                                        GenerationStarted,
                                        GenerationCompleted,
                                        GenerationCancelled,
                                        StageChanged,
                                        ErrorEvent,
                                    ),
                                )

                                # For text deltas, check if we should redraw
                                should_redraw = force_redraw
                                if isinstance(event, AssistantTextDelta):
                                    now = time.time()
                                    # Redraw if: contains newline OR enough time has passed
                                    if "\n" in event.text or (now - last_render_at) >= min_interval:
                                        should_redraw = True

                                # Update display only when needed
                                if should_redraw:
                                    live.update(build_renderable())
                                    last_render_at = time.time()

                            except asyncio.TimeoutError:
                                # No event - let Live's refresh_per_second handle spinner animation
                                # Do NOT call live.update() here to avoid flicker
                                pass
                            except asyncio.CancelledError:
                                cancelled = True
                                self.turn_state = reduce_event(
                                    self.turn_state,
                                    GenerationCancelled(),
                                )
                                live.update(build_renderable())
                                break

                        # Final render to ensure last state is displayed
                        live.update(build_renderable())

                    if cancelled:
                        self.console.print("[yellow]Generation cancelled.[/yellow]")

                finally:
                    self._in_live = False
                    # Clean up input context
                    if input_obj:
                        try:
                            input_obj.detach()
                        except Exception:
                            pass  # Ignore cleanup errors
                    if input_context:
                        try:
                            input_context.__exit__(None, None, None)
                        except Exception:
                            pass  # Ignore cleanup errors
                    if input_obj:
                        try:
                            input_obj.close()
                        except Exception:
                            pass  # Ignore cleanup errors
                    # Ensure cursor is visible after Live
                    try:
                        self.console.show_cursor(True)
                    except Exception:
                        pass
            else:
                # Fallback mode: simple line-based output
                print("\n[Assistant]")
                while not self.turn_state.completed:
                    try:
                        event = await asyncio.wait_for(
                            self.event_queue.get(),
                            timeout=0.1,
                        )
                        if event is None:
                            break

                        # Update state
                        self.turn_state = reduce_event(self.turn_state, event)
                        self.viewmodel.on_event(event)

                        # Simple output for fallback mode
                        if isinstance(event, AssistantTextDelta):
                            print(event.text, end="", flush=True)
                        elif isinstance(event, ToolCallIssued):
                            print(f"\n[Tool: {event.tool_name}]", flush=True)
                        elif isinstance(event, ToolCallFinished):
                            status = "✓" if not event.is_error else "✗"
                            print(f" {status}", flush=True)

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        self.turn_state = reduce_event(
                            self.turn_state,
                            GenerationCancelled(),
                        )
                        break
                print()  # Final newline

            # Show error if any
            if self.turn_state.last_error:
                if self._is_interactive:
                    self.console.print(f"[red]Error: {self.turn_state.last_error}[/red]")
                else:
                    print(f"Error: {self.turn_state.last_error}")

        finally:
            self._current_task = None

    def _cancel_current(self, *, quiet: bool = False) -> None:
        """Cancel current generation."""
        if self._cancelling:
            return
        self._cancelling = True
        try:
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
            if self._cancel_generation:
                self._cancel_generation()
        finally:
            self._cancelling = False

        if quiet or (self._is_interactive and self._in_live):
            return
        if self._is_interactive:
            self.console.print("[yellow]Generation cancelled.[/yellow]")
        else:
            print("Generation cancelled.")

    def _handle_slash_command(self, text: str) -> bool:
        """Handle slash command. Returns True if should exit."""
        parts = text[1:].split()
        command = parts[0] if parts else ""

        if command in {"exit", "quit"}:
            return True

        if command == "help":
            help_text = (
                "lazydspy - DSPy 优化脚本生成器\n"
                "通过对话生成可运行的 DSPy 优化脚本。\n\n"
                "Commands:\n"
                "  /help   显示帮助与说明\n"
                "  /clear  清空当前对话显示\n"
                "  /export 导出会话到 generated/<session_id>/session.json\n"
                "  /status 显示当前状态栏信息\n"
                "  /reset  清空对话并重置阶段\n"
                "  /exit   退出\n\n"
                "Shortcuts:\n"
                "  Enter         发送消息\n"
                "  Ctrl+J        插入换行\n"
                "  Alt+Enter     插入换行\n"
                "  Esc           取消生成\n"
                "  Ctrl+C        取消生成 / 退出\n"
                "  Ctrl+D        退出"
            )
            if self._is_interactive:
                self.console.print(
                    Panel(
                        help_text,
                        title="Help",
                        border_style="green",
                    )
                )
            else:
                print("\n" + "=" * 60)
                print(help_text)
                print("=" * 60 + "\n")
            return False

        if command == "clear":
            if self._is_interactive:
                self.console.clear()
            else:
                print("\n" * 50)  # Simple clear for fallback mode
            self.viewmodel.clear()
            return False

        if command == "reset":
            if self._is_interactive:
                self.console.clear()
            else:
                print("\n" * 50)
            self.viewmodel.reset()
            self.turn_count = 0
            self.turn_state = TurnState()
            if self._is_interactive:
                self.console.print("[dim]Session reset.[/dim]")
            else:
                print("Session reset.")
            return False

        if command == "status":
            status_text = (
                f"Stage: {self.turn_state.stage.value}\n"
                f"Model: {self.model}\n"
                f"Turn: {self.turn_count}\n"
                f"Generating: {self.turn_state.generating}"
            )
            if self._is_interactive:
                status = StatusLine(
                    self.turn_state,
                    self.model,
                    str(self.workdir),
                    self.turn_count,
                )
                self.console.print(status)
            else:
                print("\n" + status_text + "\n")
            return False

        if command == "export":
            path = export_session_json(self.viewmodel, self.session_dir)
            if self._is_interactive:
                self.console.print(f"[green]Session exported to {path}[/green]")
            else:
                print(f"Session exported to {path}")
            return False

        if self._is_interactive:
            self.console.print(f"[yellow]Unknown command: /{command}[/yellow]")
        else:
            print(f"Unknown command: /{command}")
        return False


__all__ = [
    "TurnState",
    "ToolCallState",
    "reduce_event",
    "ToolsBulletList",
    "AssistantBlock",
    "StatusLine",
    "LiveChatUI",
]
