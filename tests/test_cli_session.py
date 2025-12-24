import json
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


def _build_config(**kwargs):
    defaults = dict(
        session_id="session-1",
        scenario="示例场景",
        input_fields=["query"],
        output_fields=["answer"],
        model_preference="claude",
        algorithm="GEPA",
        hyperparameters={},
        data_path=None,
        mode="quick",
        subset_size=None,
        checkpoint_needed=False,
        checkpoint_dir=pathlib.Path("checkpoints"),
        checkpoint_interval=2,
        max_checkpoints=3,
        resume=False,
        generate_sample_data=False,
    )
    defaults.update(kwargs)
    return cli.GenerationConfig(**defaults)


def test_render_files_builds_data_guide(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """应写入数据指引并在结果中返回路径。"""

    monkeypatch.chdir(tmp_path)
    config = _build_config(
        scenario="问答任务",
        input_fields=["query", "context"],
        output_fields=["answer"],
    )

    result = cli._render_files(config)

    assert result.data_guide_path.exists()
    content = result.data_guide_path.read_text(encoding="utf-8")
    assert "问答任务" in content
    assert "`query`" in content
    assert "`answer`" in content
    assert result.sample_data_path is None


def test_render_files_creates_sample_data_when_enabled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """开启 generate_sample_data 时应输出样例 JSONL。"""

    monkeypatch.chdir(tmp_path)
    config = _build_config(generate_sample_data=True, input_fields=[], output_fields=["label"])

    result = cli._render_files(config)

    assert result.sample_data_path is not None
    assert result.sample_data_path.exists()
    rows = [
        json.loads(line)
        for line in result.sample_data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows and rows[0].get("label")
