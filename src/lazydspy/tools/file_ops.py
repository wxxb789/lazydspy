"""File operation tools for lazydspy Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"写入文件失败: {e}",
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
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"读取文件失败: {e}",
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
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"创建目录失败: {e}",
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
