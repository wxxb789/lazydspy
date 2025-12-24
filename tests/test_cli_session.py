import pathlib
import sys
import types

import pytest

# 提前注入 anthropic stub，避免环境缺失依赖导致导入失败。
_anthropic_stub = types.ModuleType("anthropic")
_anthropic_stub.Anthropic = object  # type: ignore[assignment]
sys.modules.setdefault("anthropic", _anthropic_stub)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from lazydspy import cli  # noqa: E402


def test_agent_session_fallback_when_client_init_fails(monkeypatch: pytest.MonkeyPatch):
    """初始化失败时，提问应回退到本地提示。"""

    def broken_client():
        raise RuntimeError("network down")

    monkeypatch.setattr(cli, "Anthropic", broken_client)

    session = cli.AgentSession(cli.console)
    prompt = session.ask("示例提示")

    assert prompt == "示例提示"


def test_agent_session_confirm_fallback(monkeypatch: pytest.MonkeyPatch):
    """模型调用异常时，确认提示应使用默认值。"""

    session = cli.AgentSession(cli.console)

    class BrokenMessages:
        def create(self, **kwargs):
            raise RuntimeError("offline")

    session._client = types.SimpleNamespace(messages=BrokenMessages())

    confirmation = session.confirm("摘要内容")

    assert confirmation == "确认上述配置并生成脚本吗？(y/N)"


def test_build_agent_session_respects_env_prompt(monkeypatch: pytest.MonkeyPatch):
    """应允许通过环境变量覆盖系统提示。"""

    monkeypatch.setenv("LAZYDSPY_SYSTEM_PROMPT", "自定义系统提示")

    session = cli._build_agent_session()

    assert session.system_prompt.startswith("自定义系统提示")


def test_agent_session_summarize_fallback(monkeypatch: pytest.MonkeyPatch):
    """无可用客户端时，应输出简短的本地摘要。"""

    session = cli.AgentSession(cli.console)
    monkeypatch.setattr(session, "_ensure_client", lambda: None)

    summary = session.summarize({"mode": "quick", "algorithm": "gepa"})

    assert "mode: quick" in summary
    assert "algorithm: gepa" in summary
