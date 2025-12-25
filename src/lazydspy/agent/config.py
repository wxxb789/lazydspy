"""Agent configuration."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for the lazydspy Agent."""

    model: str = Field(
        default="claude-opus-4.5",
        description="Claude model to use",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom API base URL",
    )
    auth_token: str | None = Field(
        default=None,
        description="API authentication token",
    )
    max_turns: int = Field(
        default=50,
        description="Maximum conversation turns",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        return cls(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4.5"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            auth_token=os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"),
            debug=os.getenv("LAZYDSPY_DEBUG", "").lower() in ("1", "true", "yes"),
        )


__all__ = ["AgentConfig"]
