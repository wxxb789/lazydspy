"""Runtime state machine for the lazydspy agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from lazydspy.specs import OptimizationSpec

if TYPE_CHECKING:
    pass


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

    _stage: ConversationStage = field(default=ConversationStage.COLLECT, init=False)
    spec: OptimizationSpec | None = None
    generated_files: list[Path] = field(default_factory=list)
    last_validation_errors: list[str] = field(default_factory=list)
    stage_change_callback: Callable[[ConversationStage], None] | None = field(
        default=None, repr=False
    )

    @property
    def stage(self) -> ConversationStage:
        """Get current conversation stage."""
        return self._stage

    @stage.setter
    def stage(self, value: ConversationStage) -> None:
        """Set conversation stage and trigger callback if registered."""
        old_stage = self._stage
        self._stage = value
        if self.stage_change_callback is not None and old_stage != value:
            self.stage_change_callback(value)


__all__ = ["AgentState", "ConversationStage"]
