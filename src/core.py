"""核心模块，包含网页摘要签名与推理封装。"""

from __future__ import annotations

import dspy

# 种子提示：指导模型在摘要时保留关键信息并保持简洁。
SEED_PROMPT: str = (
    "你是一名擅长信息提炼的中文助手，阅读网页内容后生成简洁摘要，"
    "突出核心事实与结论，避免无关细节。"
)


class WebSummarySignature(dspy.Signature):
    """网页摘要的签名定义，限定输入输出字段。"""

    content: dspy.InputField = dspy.InputField(desc="网页原始内容")
    summary: dspy.OutputField = dspy.OutputField(desc="基于链式思考生成的中文摘要")


class DeepSummarizer(dspy.Module):
    """基于链式思考的网页摘要器。"""

    def __init__(self) -> None:
        super().__init__()
        # 将链式思考模块与签名绑定。
        self._cot = dspy.ChainOfThought(WebSummarySignature)

    def forward(self, content: str) -> str:
        """执行摘要生成流程。"""
        # 调用链式思考并返回摘要文本。
        result = self._cot(content=content)
        return str(result.summary)


__all__ = ["DeepSummarizer", "SEED_PROMPT", "WebSummarySignature"]
