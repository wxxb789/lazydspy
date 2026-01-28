"""Rich Live TUI for lazydspy chat."""

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
from lazydspy.tui.live_ui import (
    AssistantBlock,
    LiveChatUI,
    StatusLine,
    ToolCallState,
    ToolsBulletList,
    TurnState,
    reduce_event,
)
from lazydspy.tui.viewmodel import Message, ToolCall, ViewModel

__all__ = [
    # Events
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
    # Live UI
    "LiveChatUI",
    "TurnState",
    "ToolCallState",
    "reduce_event",
    "ToolsBulletList",
    "AssistantBlock",
    "StatusLine",
    # ViewModel
    "Message",
    "ToolCall",
    "ViewModel",
]
