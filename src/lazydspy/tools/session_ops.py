"""Session control tools for lazydspy Agent."""

from __future__ import annotations

from typing import Any, NoReturn


class SessionComplete(Exception):
    """Raised when agent signals session completion."""

    def __init__(self, summary: str, next_steps: list[str] | None = None):
        self.summary = summary
        self.next_steps = next_steps or []
        super().__init__(summary)


async def finish_session(args: dict[str, Any]) -> NoReturn:
    """Signal that the session is complete.

    This tool should be called when:
    - All scripts have been generated and saved
    - The user has confirmed completion
    - No more actions are needed

    Args:
        args: Tool arguments containing:
            - summary: A brief summary of what was accomplished
            - next_steps: Optional list of suggested next steps

    Raises:
        SessionComplete: Always raised to signal session completion
    """
    summary = args.get("summary", "任务已完成")
    next_steps = args.get("next_steps", [])

    raise SessionComplete(summary=summary, next_steps=next_steps)


FINISH_SESSION_TOOL = {
    "name": "finish_session",
    "description": (
        "标记当前会话任务已完成。当你已经生成并保存了所有脚本文件，"
        "并且用户确认完成或没有其他需求时，调用此工具。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "任务完成摘要，描述生成了什么内容",
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "建议用户的下一步操作列表",
            },
        },
        "required": ["summary"],
    },
    "handler": finish_session,
}

SESSION_OPS_TOOLS = [FINISH_SESSION_TOOL]

__all__ = ["finish_session", "SessionComplete", "FINISH_SESSION_TOOL", "SESSION_OPS_TOOLS"]
