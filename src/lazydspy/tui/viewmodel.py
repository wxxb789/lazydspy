"""View model for TUI rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lazydspy.state import ConversationStage
from lazydspy.tui.events import (
    AssistantMessageEnd,
    AssistantMessageStart,
    AssistantTextDelta,
    ErrorEvent,
    Event,
    EventSink,
    GenerationCancelled,
    GenerationCompleted,
    GenerationStarted,
    StageChanged,
    SystemNote,
    ToolCallFinished,
    ToolCallIssued,
    UserMessageSubmitted,
)

if TYPE_CHECKING:
    from lazydspy.tui.render import MarkdownRenderer


@dataclass
class Message:
    message_id: str
    role: Literal["user", "assistant", "system"]
    text: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolCall:
    call_id: str
    name: str
    input_text: str = ""
    status: Literal["running", "ok", "error"] = "running"
    result_summary: str | None = None


@dataclass
class ViewModel(EventSink):
    session_id: str
    model: str
    workdir: Path
    stage: ConversationStage = ConversationStage.COLLECT
    generating: bool = False
    last_error: str | None = None
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    _message_index: dict[str, Message] = field(default_factory=dict, init=False)
    _tool_index: dict[str, ToolCall] = field(default_factory=dict, init=False)
    _message_counter: int = field(default=0, init=False)

    def on_event(self, event: Event) -> None:
        if isinstance(event, UserMessageSubmitted):
            self._append_message("user", event.text)
            return
        if isinstance(event, AssistantMessageStart):
            self._ensure_assistant_message(event.message_id)
            return
        if isinstance(event, AssistantTextDelta):
            message = self._ensure_assistant_message(event.message_id)
            message.text += event.text
            return
        if isinstance(event, AssistantMessageEnd):
            self._ensure_assistant_message(event.message_id)
            return
        if isinstance(event, ToolCallIssued):
            tool = self._ensure_tool_call(event.call_id, event.tool_name)
            if event.input_text:
                tool.input_text = event.input_text
            elif event.input_delta:
                tool.input_text += event.input_delta
            return
        if isinstance(event, ToolCallFinished):
            tool = self._tool_index.get(event.call_id) or self._ensure_tool_call(
                event.call_id, "unknown"
            )
            tool.status = "error" if event.is_error else "ok"
            tool.result_summary = event.result_summary
            return
        if isinstance(event, StageChanged):
            self.stage = event.stage
            return
        if isinstance(event, GenerationStarted):
            self.generating = True
            return
        if isinstance(event, GenerationCompleted):
            self.generating = False
            return
        if isinstance(event, GenerationCancelled):
            self.generating = False
            self._append_message("system", "Generation cancelled.")
            return
        if isinstance(event, ErrorEvent):
            self.last_error = event.message
            self._append_message("system", event.message)
            return
        if isinstance(event, SystemNote):
            self._append_message("system", event.text)

    def clear(self) -> None:
        self.messages = []
        self.tool_calls = []
        self._message_index = {}
        self._tool_index = {}
        self._message_counter = 0
        self.last_error = None

    def reset(self) -> None:
        self.clear()
        self.stage = ConversationStage.COLLECT
        self.generating = False

    def render_messages(self, renderer: "MarkdownRenderer", width: int) -> str:
        blocks: list[str] = []
        for message in self.messages:
            label = {
                "user": "You",
                "assistant": "Assistant",
                "system": "System",
            }.get(message.role, "Message")
            header = f"{label}:\n"
            if message.role == "assistant":
                body = renderer.render_markdown(message.text, width)
            else:
                body = renderer.render_plain(message.text)
            blocks.append((header + body).rstrip())
        return "\n\n".join(blocks)

    def render_tool_panel(self, width: int) -> str:
        lines: list[str] = []
        for tool in self.tool_calls:
            status = {
                "running": "…",
                "ok": "OK",
                "error": "ERR",
            }[tool.status]
            line = f"{status} {tool.name}"
            if tool.input_text:
                line += f" {tool.input_text.strip()}"
            if tool.result_summary:
                line += f" → {tool.result_summary.strip()}"
            lines.append(_truncate_line(line, width))
        return "\n".join(lines)

    def render_status_bar(self) -> str:
        turn_count = len([m for m in self.messages if m.role == "user"])
        stage_text = self.stage.value
        gen_text = "generating" if self.generating else "idle"
        return (
            f"Stage: {stage_text} | {gen_text} | Model: {self.model} | "
            f"Workdir: {self.workdir} | Turn: {turn_count}"
        )

    def to_session_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "workdir": str(self.workdir),
            "stage": self.stage.value,
            "generating": self.generating,
            "messages": [
                {
                    "id": message.message_id,
                    "role": message.role,
                    "text": message.text,
                    "created_at": message.created_at.isoformat(),
                }
                for message in self.messages
            ],
            "tools": [
                {
                    "id": tool.call_id,
                    "name": tool.name,
                    "input": tool.input_text,
                    "status": tool.status,
                    "result": tool.result_summary,
                }
                for tool in self.tool_calls
            ],
            "last_error": self.last_error,
        }

    def _append_message(self, role: Literal["user", "assistant", "system"], text: str) -> None:
        message_id = self._next_message_id(role)
        message = Message(message_id=message_id, role=role, text=text)
        self.messages.append(message)
        self._message_index[message_id] = message

    def _ensure_assistant_message(self, message_id: str) -> Message:
        message = self._message_index.get(message_id)
        if message is None:
            message = Message(message_id=message_id, role="assistant", text="")
            self.messages.append(message)
            self._message_index[message_id] = message
        return message

    def _ensure_tool_call(self, call_id: str, name: str) -> ToolCall:
        tool = self._tool_index.get(call_id)
        if tool is None:
            tool = ToolCall(call_id=call_id, name=name)
            self.tool_calls.append(tool)
            self._tool_index[call_id] = tool
        return tool

    def _next_message_id(self, role: str) -> str:
        self._message_counter += 1
        return f"{role}-{self._message_counter}"


def _truncate_line(text: str, width: int) -> str:
    if width <= 0:
        return text
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: max(0, width - 1)] + "…"


__all__ = ["Message", "ToolCall", "ViewModel"]
