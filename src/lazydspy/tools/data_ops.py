"""Data operation tools for lazydspy Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


async def validate_jsonl(args: dict[str, Any]) -> dict[str, Any]:
    """Validate a JSONL file format and required fields.

    Args:
        path: Path to the JSONL file
        required_fields: List of required field names
        max_lines: Maximum number of lines to check (default: 100)

    Returns:
        Tool response with validation results
    """
    try:
        path = Path(args["path"])
        required_fields = args.get("required_fields", [])
        max_lines = args.get("max_lines", 100)

        if not path.exists():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"文件不存在: {path}",
                    }
                ]
            }

        errors: list[str] = []
        valid_count = 0
        sample_row: dict[str, Any] | None = None

        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if idx > max_lines:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if sample_row is None:
                        sample_row = row

                    for field in required_fields:
                        if field not in row:
                            errors.append(f"行 {idx}: 缺少字段 '{field}'")

                    valid_count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"行 {idx}: JSON 解析错误 - {e}")

        if errors:
            error_summary = "\n".join(errors[:10])
            if len(errors) > 10:
                error_summary += f"\n... 还有 {len(errors) - 10} 个错误"

            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"验证失败，发现 {len(errors)} 个问题:\n\n{error_summary}",
                    }
                ]
            }

        # Build success message with sample
        sample_text = ""
        if sample_row:
            sample_text = f"\n\n样例数据:\n{json.dumps(sample_row, ensure_ascii=False, indent=2)}"

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"验证通过！共 {valid_count} 行有效数据。{sample_text}",
                }
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"验证失败: {e}",
                }
            ]
        }


async def check_schema(args: dict[str, Any]) -> dict[str, Any]:
    """Check if data contains expected fields.

    Args:
        data: Dictionary to check
        expected_fields: List of expected field names

    Returns:
        Tool response with check results
    """
    try:
        data = args["data"]
        expected_fields = args["expected_fields"]

        if not isinstance(data, dict):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "数据必须是一个字典",
                    }
                ]
            }

        missing = [f for f in expected_fields if f not in data]
        extra = [f for f in data if f not in expected_fields]
        present = [f for f in expected_fields if f in data]

        result_lines = ["检查结果:"]
        result_lines.append(f"- 期望字段: {expected_fields}")
        result_lines.append(f"- 已有字段: {present}")

        if missing:
            result_lines.append(f"- 缺少字段: {missing}")

        if extra:
            result_lines.append(f"- 额外字段: {extra}")

        if missing:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "\n".join(result_lines) + "\n\nSchema 检查失败：缺少必需字段",
                    }
                ]
            }

        return {
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(result_lines) + "\n\nSchema 检查通过！",
                }
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Schema 检查失败: {e}",
                }
            ]
        }


async def sample_data(args: dict[str, Any]) -> dict[str, Any]:
    """Generate sample JSONL data based on field definitions.

    Args:
        input_fields: List of input field names
        output_fields: List of output field names
        num_samples: Number of samples to generate (default: 2)
        scenario: Optional scenario description for better samples

    Returns:
        Tool response with generated sample data
    """
    try:
        input_fields = args["input_fields"]
        output_fields = args["output_fields"]
        num_samples = args.get("num_samples", 2)
        # scenario is reserved for future AI-generated sample content
        _ = args.get("scenario", "")

        samples = []

        for i in range(1, num_samples + 1):
            row: dict[str, str] = {}
            for field in input_fields:
                row[field] = f"[输入示例 {i}] {field} 的值"
            for field in output_fields:
                row[field] = f"[输出示例 {i}] {field} 的值"
            samples.append(row)

        # Generate JSONL content
        jsonl_lines = [json.dumps(row, ensure_ascii=False) for row in samples]
        jsonl_content = "\n".join(jsonl_lines)

        # Build response
        result = f"""生成了 {num_samples} 条样例数据

字段说明:
- 输入字段: {input_fields}
- 输出字段: {output_fields}

JSONL 内容 (可直接保存为 train.jsonl):
```jsonl
{jsonl_content}
```

请将示例值替换为实际的训练数据。每行一个 JSON 对象，包含所有输入和输出字段。"""

        return {
            "content": [
                {
                    "type": "text",
                    "text": result,
                }
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"生成样例数据失败: {e}",
                }
            ]
        }


# Tool definitions for MCP server registration
VALIDATE_JSONL_TOOL = {
    "name": "validate_jsonl",
    "description": "验证 JSONL 文件的格式和字段是否符合要求",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "JSONL 文件路径",
            },
            "required_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "必需的字段名列表",
                "default": [],
            },
            "max_lines": {
                "type": "integer",
                "description": "最多检查的行数",
                "default": 100,
            },
        },
        "required": ["path"],
    },
    "handler": validate_jsonl,
}

CHECK_SCHEMA_TOOL = {
    "name": "check_schema",
    "description": "检查给定数据是否包含期望的字段",
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "待检查的数据字典",
            },
            "expected_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "期望的字段名列表",
            },
        },
        "required": ["data", "expected_fields"],
    },
    "handler": check_schema,
}

SAMPLE_DATA_TOOL = {
    "name": "sample_data",
    "description": "根据字段定义生成 JSONL 样例数据",
    "input_schema": {
        "type": "object",
        "properties": {
            "input_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "输入字段名列表",
            },
            "output_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "输出字段名列表",
            },
            "num_samples": {
                "type": "integer",
                "description": "生成的样例数量",
                "default": 2,
            },
            "scenario": {
                "type": "string",
                "description": "场景描述（可选）",
                "default": "",
            },
        },
        "required": ["input_fields", "output_fields"],
    },
    "handler": sample_data,
}

DATA_OPS_TOOLS = [VALIDATE_JSONL_TOOL, CHECK_SCHEMA_TOOL, SAMPLE_DATA_TOOL]

__all__ = [
    "validate_jsonl",
    "check_schema",
    "sample_data",
    "DATA_OPS_TOOLS",
    "VALIDATE_JSONL_TOOL",
    "CHECK_SCHEMA_TOOL",
    "SAMPLE_DATA_TOOL",
]
