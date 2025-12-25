"""Conversation session management."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# Type alias for API message format
# content can be str or list[dict] (for tool_use/tool_result)
APIMessage = dict[str, Any]


class Message(BaseModel):
    """A conversation message."""

    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Message timestamp",
    )


class SessionState(BaseModel):
    """State of a conversation session."""

    session_id: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        description="Unique session identifier",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="Conversation history",
    )
    tool_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Results from tool calls",
    )


class ConversationSession:
    """Manages a conversation session with the Agent."""

    def __init__(self, session_id: str | None = None) -> None:
        """Initialize a new conversation session.

        Args:
            session_id: Optional custom session ID
        """
        self.state = SessionState()
        if session_id:
            self.state.session_id = session_id

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self.state.session_id

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.

        Args:
            content: The message content
        """
        self.state.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.

        Args:
            content: The message content
        """
        self.state.messages.append(Message(role="assistant", content=content))

    def add_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Record a tool execution result.

        Args:
            tool_name: Name of the tool
            result: The tool's output
        """
        self.state.tool_results.append({
            "tool": tool_name,
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        })

    def get_messages(self) -> list[APIMessage]:
        """Get messages in API format.

        Returns:
            List of message dicts with role and content
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.state.messages]

    def get_last_assistant_message(self) -> str | None:
        """Get the last assistant message.

        Returns:
            The content of the last assistant message, or None
        """
        for msg in reversed(self.state.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def message_count(self) -> int:
        """Get the total number of messages.

        Returns:
            Number of messages in the conversation
        """
        return len(self.state.messages)

    def clear(self) -> None:
        """Clear all messages and tool results."""
        self.state.messages.clear()
        self.state.tool_results.clear()


__all__ = ["Message", "SessionState", "ConversationSession", "APIMessage"]
