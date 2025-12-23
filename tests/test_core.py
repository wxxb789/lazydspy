import types

import dspy
import pytest

from core import DeepSummarizer, SEED_PROMPT, WebSummarySignature


def test_web_summary_signature_fields():
    """签名应暴露输入输出字段，并保留种子提示文本。"""
    assert isinstance(WebSummarySignature.content, dspy.InputField)
    assert isinstance(WebSummarySignature.summary, dspy.OutputField)
    assert "摘要" in SEED_PROMPT


def test_deep_summarizer_forward_returns_summary(monkeypatch: pytest.MonkeyPatch):
    """forward 应调用链式思考并返回字符串摘要。"""
    call_args = {}

    class DummyChain:
        def __call__(self, *, content: str):
            call_args["content"] = content
            return types.SimpleNamespace(summary="模拟摘要")

    def fake_chain(signature):
        call_args["signature"] = signature
        return DummyChain()

    monkeypatch.setattr(dspy, "ChainOfThought", fake_chain)

    summarizer = DeepSummarizer()
    output = summarizer.forward("示例内容")

    assert output == "模拟摘要"
    assert call_args["content"] == "示例内容"
    assert call_args["signature"] is WebSummarySignature

