"""File operation tools for lazydspy Agent."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _format_exception_details(e: Exception, context: dict[str, Any] | None = None) -> str:
    """Format detailed exception information for debugging."""

    def _add_section(title: str | None, lines: list[str]) -> None:
        clean = [line for line in lines if line is not None]
        if not clean:
            return
        if title:
            sections.append(f"{title}\n" + "\n".join(clean))
        else:
            sections.append("\n".join(clean))

    sections: list[str] = []

    # Basic exception info
    basic_info = [
        f"异常类型: {type(e).__module__}.{type(e).__name__}",
        f"异常消息: {e}",
        f"异常参数: {e.args}",
    ]

    # OSError specific attributes
    if isinstance(e, OSError):
        basic_info.extend(
            f"{name}: {value}"
            for name, value in (
                ("错误码 (errno)", e.errno),
                ("系统错误消息", e.strerror),
                ("相关文件", e.filename),
                ("相关文件2", e.filename2),
            )
            if value is not None
        )

    # Exception chain
    if e.__cause__:
        basic_info.append(f"直接原因 (__cause__): {type(e.__cause__).__name__}: {e.__cause__}")
    if e.__context__ and e.__context__ is not e.__cause__:
        basic_info.append(f"上下文异常 (__context__): {type(e.__context__).__name__}: {e.__context__}")

    _add_section(None, basic_info)

    # Context information
    if context:
        context_lines = [f"{k}: {v}" for k, v in context.items() if v is not None]
        _add_section("=== 上下文信息 ===", context_lines)

    # Environment info
    env_info = [
        f"当前工作目录: {os.getcwd()}",
        f"Python 版本: {sys.version}",
        f"平台: {sys.platform}",
    ]
    _add_section("=== 环境信息 ===", env_info)

    # Full traceback (includes cause/context automatically)
    tb_text = "".join(traceback.TracebackException.from_exception(e).format())
    _add_section("=== 完整堆栈跟踪 ===", [tb_text])

    return "\n\n".join(sections)


async def write_file(args: dict[str, Any]) -> dict[str, Any]:
    """Write content to a file at the specified path.

    Args:
        path: File path to write to
        content: Content to write
        encoding: File encoding (default: utf-8)

    Returns:
        Tool response with success status
    """
    try:
        path = Path(args["path"])
        content = args["content"]
        encoding = args.get("encoding", "utf-8")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        path.write_text(content, encoding=encoding)

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"文件已成功写入: {path}\n文件大小: {len(content)} 字符",
                }
            ]
        }
    except Exception as e:
        context = {
            "操作": "write_file",
            "目标路径": args.get("path"),
            "绝对路径": str(Path(args.get("path", "")).absolute()) if args.get("path") else None,
            "内容长度": len(args.get("content", "")) if args.get("content") else None,
            "编码": args.get("encoding", "utf-8"),
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"写入文件失败\n\n{_format_exception_details(e, context)}",
                }
            ]
        }


async def read_file(args: dict[str, Any]) -> dict[str, Any]:
    """Read content from a file.

    Args:
        path: File path to read from
        encoding: File encoding (default: utf-8)

    Returns:
        Tool response with file content
    """
    try:
        path = Path(args["path"])
        encoding = args.get("encoding", "utf-8")

        if not path.exists():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"文件不存在: {path}",
                    }
                ]
            }

        content = path.read_text(encoding=encoding)

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"文件内容 ({path}):\n\n{content}",
                }
            ]
        }
    except Exception as e:
        context = {
            "操作": "read_file",
            "目标路径": args.get("path"),
            "绝对路径": str(Path(args.get("path", "")).absolute()) if args.get("path") else None,
            "编码": args.get("encoding", "utf-8"),
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"读取文件失败\n\n{_format_exception_details(e, context)}",
                }
            ]
        }


async def create_dir(args: dict[str, Any]) -> dict[str, Any]:
    """Create a directory.

    Args:
        path: Directory path to create
        parents: Whether to create parent directories (default: true)

    Returns:
        Tool response with success status
    """
    try:
        path = Path(args["path"])
        parents = args.get("parents", True)

        path.mkdir(parents=parents, exist_ok=True)

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"目录已创建: {path}",
                }
            ]
        }
    except Exception as e:
        context = {
            "操作": "create_dir",
            "目标路径": args.get("path"),
            "绝对路径": str(Path(args.get("path", "")).absolute()) if args.get("path") else None,
            "创建父目录": args.get("parents", True),
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"创建目录失败\n\n{_format_exception_details(e, context)}",
                }
            ]
        }


# Tool definitions for MCP server registration
WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "将内容写入指定路径的文件。如果目录不存在会自动创建。",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
            "content": {
                "type": "string",
                "description": "要写入的内容",
            },
            "encoding": {
                "type": "string",
                "description": "文件编码，默认 utf-8",
                "default": "utf-8",
            },
        },
        "required": ["path", "content"],
    },
    "handler": write_file,
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "读取指定路径的文件内容",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
            "encoding": {
                "type": "string",
                "description": "文件编码，默认 utf-8",
                "default": "utf-8",
            },
        },
        "required": ["path"],
    },
    "handler": read_file,
}

CREATE_DIR_TOOL = {
    "name": "create_dir",
    "description": "创建目录。如果父目录不存在会自动创建。",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径",
            },
            "parents": {
                "type": "boolean",
                "description": "是否创建父目录，默认 true",
                "default": True,
            },
        },
        "required": ["path"],
    },
    "handler": create_dir,
}

FILE_OPS_TOOLS = [WRITE_FILE_TOOL, READ_FILE_TOOL, CREATE_DIR_TOOL]

__all__ = [
    "write_file",
    "read_file",
    "create_dir",
    "FILE_OPS_TOOLS",
    "WRITE_FILE_TOOL",
    "READ_FILE_TOOL",
    "CREATE_DIR_TOOL",
]
