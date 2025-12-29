"""Runtime state machine for the lazydspy agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lazydspy.specs import OptimizationSpec


class ConversationStage(str, Enum):
    """Conversation lifecycle stages."""

    COLLECT = "collect"
    CONFIRM = "confirm"
    GENERATE = "generate"
    VALIDATE = "validate"
    DONE = "done"


@dataclass
class AgentState:
    """Mutable state for a single agent session."""

    stage: ConversationStage = ConversationStage.COLLECT
    spec: OptimizationSpec | None = None
    generated_files: list[Path] = field(default_factory=list)
    last_validation_errors: list[str] = field(default_factory=list)


__all__ = ["AgentState", "ConversationStage"]
