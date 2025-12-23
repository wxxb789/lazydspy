"""轻量 dspy 占位实现，用于本地测试与类型引用。"""

from __future__ import annotations

from typing import Any


class InputField:
    """输入字段占位符。"""

    def __init__(self, desc: str | None = None) -> None:
        self.desc = desc or ""


class OutputField:
    """输出字段占位符。"""

    def __init__(self, desc: str | None = None) -> None:
        self.desc = desc or ""


class Signature:
    """签名基类，无额外行为。"""


class ChainOfThought:
    """链式思考占位实现。"""

    def __init__(self, signature: type[Signature]) -> None:
        self.signature = signature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ChainOfThought placeholder cannot run directly.")


class Module:
    """模块基类，占位用于继承。"""

    def __init__(self) -> None:
        ...


__all__ = ["ChainOfThought", "InputField", "Module", "OutputField", "Signature"]
