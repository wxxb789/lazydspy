import json
import pathlib
import sys
import types
from typing import Any, Callable

import pytest

# 注入假 openai 模块，避免环境缺少依赖时导入失败。
_openai_stub = types.ModuleType("openai")
_openai_stub.OpenAI = object  # type: ignore[assignment]
sys.modules.setdefault("openai", _openai_stub)

# 确保 src 目录在导入路径中，便于直接导入 metrics 模块。
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import metrics  # noqa: E402  # isort: skip


class _FakeCompletions:
    """模拟 OpenAI completions 接口。"""

    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self._content))]
        )


class _FakeChat:
    """模拟 OpenAI chat 字段。"""

    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeOpenAI:
    """可配置返回内容的 OpenAI 替身。"""

    def __init__(self, content: str) -> None:
        self._content = content
        self.chat = _FakeChat(content)


def _patch_openai(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    """统一替换 OpenAI 客户端，返回指定内容。"""

    factory: Callable[[], _FakeOpenAI] = lambda: _FakeOpenAI(content)
    monkeypatch.setattr(metrics, "OpenAI", factory)


def test_llm_judge_metric_returns_normalized_score(monkeypatch: pytest.MonkeyPatch):
    """当 LLM 返回有效 JSON 时，应按规则归一化得分。"""

    payload = {
        "score": 4.0,  # 总分 4 => 0.8
        "details": {
            "relevance": 5,
            "accuracy": 5,
            "completeness": 5,
            "clarity": 5,
            "formatting": 5,
        },
        "feedback": "很好",
    }
    content = json.dumps(payload)

    _patch_openai(monkeypatch, content)

    score = metrics.llm_judge_metric(gold="参考", pred="模型答案")

    # detail_avg=5 => 1.0，overall=4 => 0.8，最终均值 0.9
    assert score == pytest.approx(0.9)


@pytest.mark.parametrize(
    "content",
    [
        "非 JSON",
        "{}",
        json.dumps({"details": {"relevance": 10}}),  # 越界字段\n",
    ],
)
def test_llm_judge_metric_invalid_response_returns_zero(
    monkeypatch: pytest.MonkeyPatch, content: str
):
    """解析失败或字段越界时，应回退 0.0。"""

    _patch_openai(monkeypatch, content)

    score = metrics.llm_judge_metric(gold="参考", pred="错误答案")

    assert score == 0.0
