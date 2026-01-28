from __future__ import annotations

import io
import sys
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console
from rich.syntax import Syntax

from lazydspy.state import ConversationStage
from lazydspy.tui import StatusLine, ToolCallState, ToolsBulletList, TurnState, reduce_event
from lazydspy.tui.completions import FileReferenceCompleter, SlashCommandCompleter
from lazydspy.tui.events import (
    AssistantTextDelta,
    ErrorEvent,
    GenerationCancelled,
    GenerationCompleted,
    GenerationStarted,
    StageChanged,
    ToolCallFinished,
    ToolCallIssued,
)
from lazydspy.tui.live_ui import _render_compact_markdown
from lazydspy.tui.render import MarkdownRenderer


def test_slash_command_completer() -> None:
    completer = SlashCommandCompleter()
    document = Document(text="/he", cursor_position=3)
    completions = list(completer.get_completions(document, CompleteEvent()))
    assert any(item.text == "/help" for item in completions)


def test_file_reference_completer(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "file.txt"
    target.write_text("ok", encoding="utf-8")

    completer = FileReferenceCompleter(workspace_root=tmp_path)
    document = Document(text="@data/f", cursor_position=len("@data/f"))
    completions = list(completer.get_completions(document, CompleteEvent()))
    assert any(item.text == "@data/file.txt" for item in completions)


def test_markdown_renderer() -> None:
    renderer = MarkdownRenderer()
    rendered = renderer.render_markdown("**bold**", width=40)
    assert "bold" in rendered


# ============================================================================
# Reducer Tests
# ============================================================================


def test_reduce_assistant_text_delta() -> None:
    """Test AssistantTextDelta accumulates text."""
    state = TurnState()
    event = AssistantTextDelta(message_id="msg1", text="Hello")
    new_state = reduce_event(state, event)
    assert new_state.assistant_buffer == "Hello"

    # Accumulates
    event2 = AssistantTextDelta(message_id="msg1", text=" World")
    new_state2 = reduce_event(new_state, event2)
    assert new_state2.assistant_buffer == "Hello World"


def test_reduce_tool_call_issued() -> None:
    """Test ToolCallIssued creates tool call entry."""
    state = TurnState()
    event = ToolCallIssued(
        call_id="call1",
        tool_name="read",
        input_text='{"filePath": "test.py"}',
    )
    new_state = reduce_event(state, event)

    assert "call1" in new_state.tool_calls
    assert new_state.tool_calls["call1"].name == "read"
    assert new_state.tool_calls["call1"].status == "running"
    assert new_state.tool_calls["call1"].input_text == '{"filePath": "test.py"}'
    assert new_state.tool_order == ["call1"]


def test_reduce_tool_call_issued_with_delta() -> None:
    """Test ToolCallIssued accumulates input_delta."""
    state = TurnState()

    # First delta creates tool call
    event1 = ToolCallIssued(call_id="call1", tool_name="read", input_delta='{"file')
    new_state = reduce_event(state, event1)
    assert new_state.tool_calls["call1"].input_text == '{"file'

    # Second delta accumulates
    event2 = ToolCallIssued(call_id="call1", tool_name="read", input_delta='Path": "test.py"}')
    new_state2 = reduce_event(new_state, event2)
    assert new_state2.tool_calls["call1"].input_text == '{"filePath": "test.py"}'


def test_reduce_tool_call_finished_ok() -> None:
    """Test ToolCallFinished updates status to ok."""
    state = TurnState(
        tool_calls={
            "call1": ToolCallState(call_id="call1", name="read", status="running"),
        },
        tool_order=["call1"],
    )

    event = ToolCallFinished(
        call_id="call1",
        result_summary="File read successfully",
        is_error=False,
    )
    new_state = reduce_event(state, event)

    assert new_state.tool_calls["call1"].status == "ok"
    assert new_state.tool_calls["call1"].result_summary == "File read successfully"


def test_reduce_tool_call_finished_error() -> None:
    """Test ToolCallFinished updates status to error."""
    state = TurnState(
        tool_calls={
            "call1": ToolCallState(call_id="call1", name="read", status="running"),
        },
        tool_order=["call1"],
    )

    event = ToolCallFinished(
        call_id="call1",
        result_summary="File not found",
        is_error=True,
    )
    new_state = reduce_event(state, event)

    assert new_state.tool_calls["call1"].status == "error"
    assert new_state.tool_calls["call1"].result_summary == "File not found"


def test_reduce_generation_started() -> None:
    """Test GenerationStarted sets generating=True."""
    state = TurnState()
    event = GenerationStarted()
    new_state = reduce_event(state, event)
    assert new_state.generating is True


def test_reduce_generation_completed() -> None:
    """Test GenerationCompleted sets generating=False, completed=True."""
    state = TurnState(generating=True)
    event = GenerationCompleted()
    new_state = reduce_event(state, event)
    assert new_state.generating is False
    assert new_state.completed is True


def test_reduce_generation_cancelled() -> None:
    """Test GenerationCancelled sets generating=False, completed=True."""
    state = TurnState(generating=True)
    event = GenerationCancelled()
    new_state = reduce_event(state, event)
    assert new_state.generating is False
    assert new_state.completed is True


def test_reduce_stage_changed() -> None:
    """Test StageChanged updates stage."""
    state = TurnState(stage=ConversationStage.COLLECT)
    event = StageChanged(stage=ConversationStage.GENERATE)
    new_state = reduce_event(state, event)
    assert new_state.stage == ConversationStage.GENERATE


def test_reduce_error_event() -> None:
    """Test ErrorEvent sets last_error."""
    state = TurnState()
    event = ErrorEvent(message="Something went wrong", recoverable=True)
    new_state = reduce_event(state, event)
    assert new_state.last_error == "Something went wrong"


def test_reduce_multiple_tool_calls() -> None:
    """Test multiple tool calls preserve order."""
    state = TurnState()

    # First tool call
    event1 = ToolCallIssued(call_id="call1", tool_name="read", input_text="file1.py")
    state = reduce_event(state, event1)

    # Second tool call
    event2 = ToolCallIssued(call_id="call2", tool_name="write", input_text="file2.py")
    state = reduce_event(state, event2)

    # Third tool call
    event3 = ToolCallIssued(call_id="call3", tool_name="edit", input_text="file3.py")
    state = reduce_event(state, event3)

    assert state.tool_order == ["call1", "call2", "call3"]
    assert len(state.tool_calls) == 3


# ============================================================================
# Renderable Tests
# ============================================================================


def test_status_line_contains_metadata() -> None:
    state = TurnState(stage=ConversationStage.COLLECT)
    status = StatusLine(state, model="test-model", workdir="/tmp/workspace", turn=3)

    console = Console(record=True, width=120, force_terminal=True)
    console.print(status)
    text = console.export_text()

    assert "Stage: collect" in text
    assert "Model: test-model" in text
    assert "Workdir: workspace" in text
    assert "Turn: 3" in text
    assert "Enter: send" in text


def test_tools_bullet_list_ascii_fallback(monkeypatch: MonkeyPatch) -> None:
    class DummyStdout(io.StringIO):
        encoding = "ascii"

    monkeypatch.setattr(sys, "stdout", DummyStdout())

    state = TurnState(
        tool_calls={
            "ok": ToolCallState(call_id="ok", name="read", status="ok"),
            "err": ToolCallState(call_id="err", name="write", status="error"),
        },
        tool_order=["ok", "err"],
    )

    console = Console(record=True, width=120, force_terminal=True)
    console.print(ToolsBulletList(state))
    text = console.export_text()

    assert "OK" in text
    assert "X" in text


def test_render_compact_markdown_codeblock_uses_minimal_syntax() -> None:
    console = Console(record=True, width=120, force_terminal=True)
    group = _render_compact_markdown("```python\nprint('x')\n```", console)

    syntax_blocks = [item for item in group.renderables if isinstance(item, Syntax)]
    assert syntax_blocks, "Expected a Syntax block for fenced code"

    syntax_block = syntax_blocks[0]
    assert syntax_block.background_color is None
    padding = syntax_block.padding
    if isinstance(padding, tuple):
        assert all(value == 0 for value in padding)
    else:
        assert padding == 0
