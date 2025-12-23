"""轻量 dspy 占位实现，用于本地测试与类型引用。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple


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


@dataclass
class _OptimizedProgram:
    """占位优化结果，携带最终提示与调试信息。"""

    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Example(dict[str, Any]):
    """示例容器，模仿 dspy.Example 的最小接口。"""

    # 将输入字段名缓存下来，便于与真实 dspy 行为对齐。
    inputs: Tuple[str, ...] = field(default_factory=tuple)

    def __init__(self, **kwargs: Any) -> None:  # noqa: D401 - 与 dict 行为保持一致
        super().__init__(**kwargs)
        # 将键值对同步到属性访问，方便点操作。
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.inputs = tuple()

    def with_inputs(self, *fields: str) -> "Example":
        """记录输入字段名称并返回自身，便于链式调用。"""

        self.inputs = tuple(fields)
        return self


class LM:
    """语言模型占位符，保存初始化参数。"""

    def __init__(self, model: str, **config: Any) -> None:
        self.model = model
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover - 纯展示逻辑
        return f"LM(model={self.model!r}, config={self.config!r})"


class OpenAI(LM):
    """OpenAI 兼容封装，继承自 :class:`LM`。"""


class _Settings:
    """全局配置存储，模拟 dspy.settings 行为。"""

    def __init__(self) -> None:
        self.lm: LM | None = None
        self.teacher_model: LM | None = None
        self.student_model: LM | None = None

    def configure(self, **kwargs: Any) -> None:
        """更新配置字段。"""

        for key, value in kwargs.items():
            setattr(self, key, value)


settings = _Settings()


def configure(**kwargs: Any) -> None:
    """模块级 configure，调用全局设置存储。"""

    settings.configure(**kwargs)


class GEvalPromptedAssembly:
    """GEPA 占位实现，保留初始化参数并构造伪最优提示。"""

    def __init__(
        self,
        metric: Any,
        prompt_model: LM,
        breadth: int,
        depth: int,
        temperature: float,
    ) -> None:
        self.metric = metric
        self.prompt_model = prompt_model
        self.breadth = breadth
        self.depth = depth
        self.temperature = temperature

    def compile(
        self,
        trainset: list[Example],
        valset: list[Example],
        seed_prompt: str,
    ) -> _OptimizedProgram:
        """返回携带伪最优提示的占位结果。"""

        tuned_prompt = (
            f"{seed_prompt}\n"
            f"（占位 GEPA 生成，深度 {self.depth}，宽度 {self.breadth}，温度 {self.temperature}）"
        )
        meta = {
            "breadth": self.breadth,
            "depth": self.depth,
            "temperature": self.temperature,
            "train_size": len(trainset),
            "val_size": len(valset),
            "prompt_model": repr(self.prompt_model),
        }
        return _OptimizedProgram(prompt=tuned_prompt, meta=meta)


__all__ = [
    "ChainOfThought",
    "Example",
    "GEvalPromptedAssembly",
    "InputField",
    "LM",
    "Module",
    "OpenAI",
    "OutputField",
    "Signature",
    "configure",
    "settings",
]
