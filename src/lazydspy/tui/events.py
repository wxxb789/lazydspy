"""Event protocol for the TUI layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

from lazydspy.state import ConversationStage


@dataclass(frozen=True)
class UserMessageSubmitted:
    text: str


@dataclass(frozen=True)
class AssistantMessageStart:
    message_id: str


@dataclass(frozen=True)
class AssistantTextDelta:
    message_id: str
    text: str


@dataclass(frozen=True)
class AssistantMessageEnd:
    message_id: str


@dataclass(frozen=True)
class ToolCallIssued:
    call_id: str
    tool_name: str
    input_delta: str | None = None
    input_text: str | None = None


@dataclass(frozen=True)
class ToolCallFinished:
    call_id: str
    result_summary: str | None = None
    is_error: bool = False


@dataclass(frozen=True)
class StageChanged:
    stage: ConversationStage


@dataclass(frozen=True)
class GenerationStarted:
    pass


@dataclass(frozen=True)
class GenerationCompleted:
    pass


@dataclass(frozen=True)
class GenerationCancelled:
    pass


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    recoverable: bool = True


@dataclass(frozen=True)
class SystemNote:
    text: str


Event = Union[
    UserMessageSubmitted,
    AssistantMessageStart,
    AssistantTextDelta,
    AssistantMessageEnd,
    ToolCallIssued,
    ToolCallFinished,
    StageChanged,
    GenerationStarted,
    GenerationCompleted,
    GenerationCancelled,
    ErrorEvent,
    SystemNote,
]


class EventSink(Protocol):
    def on_event(self, event: Event) -> None:
        """Consume an event."""


__all__ = [
    "AssistantMessageEnd",
    "AssistantMessageStart",
    "AssistantTextDelta",
    "ErrorEvent",
    "Event",
    "EventSink",
    "GenerationCancelled",
    "GenerationCompleted",
    "GenerationStarted",
    "StageChanged",
    "SystemNote",
    "ToolCallFinished",
    "ToolCallIssued",
    "UserMessageSubmitted",
]
