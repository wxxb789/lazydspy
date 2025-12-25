import json
import os
import pathlib
import sys
import types

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from lazydspy import GEPA_PRESETS, MIPROV2_PRESETS, cli  # noqa: E402


def test_agent_session_fallback_when_client_init_fails(monkeypatch: pytest.MonkeyPatch):
    """初始化失败时，提问应回退到本地提示。"""

    def broken_session(_self):
        raise RuntimeError("network down")

    monkeypatch.setattr(cli.AgentSession, "_build_sdk_session", broken_session)

    session = cli.AgentSession(cli.console)
    prompt = session.ask("示例提示")

    assert prompt == "示例提示"


def test_agent_session_confirm_fallback(monkeypatch: pytest.MonkeyPatch):
    """模型调用异常时，确认提示应使用默认值。"""

    session = cli.AgentSession(cli.console)

    class BrokenMessages:
        def create(self, **kwargs):
            raise RuntimeError("offline")

    session._session = types.SimpleNamespace(messages=BrokenMessages())

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
    monkeypatch.setattr(session, "_ensure_session", lambda: None)

    summary = session.summarize({"mode": "quick", "algorithm": "gepa"})

    assert "mode: quick" in summary
    assert "algorithm: gepa" in summary


def test_build_agent_session_respects_model_env(monkeypatch: pytest.MonkeyPatch):
    """应允许通过环境变量或参数覆盖默认模型。"""

    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    session = cli._build_agent_session()

    assert session._model == "claude-3-5-haiku-latest"


def test_build_agent_session_warns_when_base_url_missing_token(
    monkeypatch: pytest.MonkeyPatch,
):
    """缺少鉴权时应给出清晰警告。"""

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:8080")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    messages: list[str] = []
    monkeypatch.setattr(cli.console, "print", lambda msg: messages.append(str(msg)))

    session = cli._build_agent_session()

    assert session._base_url == "http://localhost:8080"
    assert any("自定义 Claude Endpoint 初始化失败" in msg for msg in messages)


def test_build_agent_session_applies_agent_profile_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    """Agent 配置文件中的 env/permissions 应应用到 SDK 初始化。"""

    profile_path = tmp_path / "agent.json"
    profile_path.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "http://localhost:4141",
                    "ANTHROPIC_AUTH_TOKEN": "dummy",
                    "ANTHROPIC_MODEL": "claude-opus-4.5",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-opus-4.5",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4.5",
                    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                },
                "permissions": {"deny": ["WebSearch"]},
            }
        ),
        encoding="utf-8",
    )

    for key in [
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    ]:
        monkeypatch.delenv(key, raising=False)

    session = cli._build_agent_session(agent_config=profile_path)

    assert os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL") == "claude-opus-4.5"
    assert os.getenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") == "1"
    assert "WebSearch" in session._denied_tools


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
        checkpoint_interval=1,
        max_checkpoints=20,
        resume=False,
        generate_sample_data=False,
    )
    defaults.update(kwargs)
    return cli.GenerationConfig(**defaults)


def test_generation_config_uses_mode_presets() -> None:
    """quick/full 模式应自动填充算法预设的超参。"""

    config = _build_config(algorithm="GEPA", mode="quick", hyperparameters={})

    assert config.active_hyperparameters["breadth"] == GEPA_PRESETS["quick"]["breadth"]
    assert config.active_hyperparameters["temperature"] == GEPA_PRESETS["quick"]["temperature"]


def test_generation_config_overrides_are_typed() -> None:
    """应接受字典覆盖并落在对应算法字段上。"""

    config = _build_config(algorithm="MIPROv2", mode="full", hyperparameters={"search_size": 5})

    assert config.hyperparameters.search_size == 5
    assert config.active_hyperparameters["temperature"] == MIPROV2_PRESETS["full"]["temperature"]


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
