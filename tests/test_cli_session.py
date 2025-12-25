"""Tests for the new Agentic architecture.

These tests cover:
- models/ - GenerationConfig and hyperparameters
- knowledge/ - Optimizers and cost models
- tools/ - Tool handlers
- agent/ - Session and config
"""

import asyncio
import json
import pathlib
import sys

import pytest

# Ensure real pydantic is used, not the src/pydantic.py stub
# by appending src path instead of inserting at the beginning
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from lazydspy import GEPA_PRESETS, MIPROV2_PRESETS  # noqa: E402
from lazydspy.models import (  # noqa: E402
    GenerationConfig,
    GEPAHyperparameters,
    MIPROv2Hyperparameters,
)
from lazydspy.knowledge import (  # noqa: E402
    estimate_optimization_cost,
    get_optimizer_info,
    list_all_optimizers,
)
from lazydspy.agent import AgentConfig, ConversationSession  # noqa: E402


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ============================================================================
# Models Tests
# ============================================================================


def _build_config(**kwargs):
    """Helper to build GenerationConfig with defaults."""
    defaults = dict(
        session_id="session-1",
        scenario="示例场景",
        input_fields=["query"],
        output_fields=["answer"],
        model_preference="claude-opus-4.5",
        algorithm="GEPA",
        hyperparameters={},
        data_path=None,
        mode="quick",
        subset_size=None,
        checkpoint_enabled=False,
        checkpoint_dir=pathlib.Path("checkpoints"),
        checkpoint_interval=1,
        max_checkpoints=20,
        resume=False,
        generate_sample_data=False,
    )
    defaults.update(kwargs)
    return GenerationConfig(**defaults)


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


def test_generation_config_normalizes_algorithm() -> None:
    """算法名称应规范化为标准格式。"""
    config1 = _build_config(algorithm="gepa")
    assert config1.algorithm == "GEPA"

    config2 = _build_config(algorithm="mipro-v2")
    assert config2.algorithm == "MIPROv2"

    config3 = _build_config(algorithm="MIPROv2")
    assert config3.algorithm == "MIPROv2"


def test_generation_config_parses_field_strings() -> None:
    """字段列表应支持逗号分隔的字符串输入。"""
    config = _build_config(input_fields="query, context", output_fields="answer, score")

    assert config.input_fields == ["query", "context"]
    assert config.output_fields == ["answer", "score"]


def test_generation_config_validates_mode() -> None:
    """模式应只接受 quick 或 full。"""
    with pytest.raises(ValueError, match="must be quick or full"):
        _build_config(mode="invalid")


def test_gepa_hyperparameters_from_mode() -> None:
    """GEPA 超参应根据模式应用预设。"""
    quick = GEPAHyperparameters.from_mode("quick")
    assert quick.breadth == 2
    assert quick.depth == 2

    full = GEPAHyperparameters.from_mode("full")
    assert full.breadth == 4
    assert full.depth == 4


def test_miprov2_hyperparameters_from_mode() -> None:
    """MIPROv2 超参应根据模式应用预设。"""
    quick = MIPROv2Hyperparameters.from_mode("quick")
    assert quick.search_size == 8

    full = MIPROv2Hyperparameters.from_mode("full")
    assert full.search_size == 16


# ============================================================================
# Knowledge Tests
# ============================================================================


def test_get_optimizer_info() -> None:
    """应能获取优化器信息。"""
    gepa = get_optimizer_info("gepa")
    assert gepa is not None
    assert gepa.name == "GEPA"
    assert "general" in gepa.recommended_for

    mipro = get_optimizer_info("miprov2")
    assert mipro is not None
    assert mipro.name == "MIPROv2"


def test_list_all_optimizers() -> None:
    """应列出所有可用优化器。"""
    optimizers = list_all_optimizers()
    assert len(optimizers) == 2

    names = [opt["key"] for opt in optimizers]
    assert "gepa" in names
    assert "miprov2" in names


def test_estimate_optimization_cost() -> None:
    """应能估算优化成本。"""
    result = estimate_optimization_cost(
        optimizer="gepa",
        mode="quick",
        dataset_size=100,
        model="claude-opus-4.5",
    )

    assert "estimated_cost_usd" in result
    assert result["estimated_cost_usd"] >= 0
    assert "cost_hint" in result
    assert "estimated_calls" in result


def test_estimate_cost_full_mode_higher() -> None:
    """full 模式的成本应高于 quick 模式。"""
    quick = estimate_optimization_cost(
        optimizer="gepa", mode="quick", dataset_size=100
    )
    full = estimate_optimization_cost(
        optimizer="gepa", mode="full", dataset_size=100
    )

    assert full["estimated_cost_usd"] > quick["estimated_cost_usd"]


# ============================================================================
# Agent Tests
# ============================================================================


def test_agent_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """AgentConfig 应从环境变量读取配置。"""
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")

    config = AgentConfig.from_env()

    assert config.model == "test-model"
    assert config.auth_token == "test-key"


def test_agent_config_defaults() -> None:
    """AgentConfig 应有合理的默认值。"""
    config = AgentConfig()

    assert config.model == "claude-opus-4.5"
    assert config.base_url is None
    assert config.max_turns == 50


def test_conversation_session_messages() -> None:
    """ConversationSession 应正确管理消息。"""
    session = ConversationSession()

    session.add_user_message("Hello")
    session.add_assistant_message("Hi there!")
    session.add_user_message("How are you?")

    messages = session.get_messages()
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"


def test_conversation_session_last_message() -> None:
    """应能获取最后一条助手消息。"""
    session = ConversationSession()

    session.add_user_message("Q1")
    session.add_assistant_message("A1")
    session.add_user_message("Q2")
    session.add_assistant_message("A2")

    last = session.get_last_assistant_message()
    assert last == "A2"


def test_conversation_session_clear() -> None:
    """应能清空会话。"""
    session = ConversationSession()

    session.add_user_message("Test")
    session.add_assistant_message("Response")

    assert session.message_count() == 2

    session.clear()

    assert session.message_count() == 0


# ============================================================================
# Tools Tests
# ============================================================================


def test_write_file_tool(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """write_file 工具应能写入文件。"""
    monkeypatch.chdir(tmp_path)

    from lazydspy.tools.file_ops import write_file

    result = run_async(write_file({
        "path": str(tmp_path / "test.txt"),
        "content": "Hello, World!",
    }))

    assert "文件已成功写入" in result["content"][0]["text"]
    assert (tmp_path / "test.txt").read_text() == "Hello, World!"


def test_read_file_tool(tmp_path: pathlib.Path) -> None:
    """read_file 工具应能读取文件。"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content", encoding="utf-8")

    from lazydspy.tools.file_ops import read_file

    result = run_async(read_file({"path": str(test_file)}))

    assert "Test content" in result["content"][0]["text"]


def test_read_file_not_found(tmp_path: pathlib.Path) -> None:
    """read_file 工具应处理文件不存在的情况。"""
    from lazydspy.tools.file_ops import read_file

    result = run_async(read_file({"path": str(tmp_path / "nonexistent.txt")}))

    assert "文件不存在" in result["content"][0]["text"]


def test_create_dir_tool(tmp_path: pathlib.Path) -> None:
    """create_dir 工具应能创建目录。"""
    from lazydspy.tools.file_ops import create_dir

    new_dir = tmp_path / "subdir" / "nested"
    result = run_async(create_dir({"path": str(new_dir)}))

    assert "目录已创建" in result["content"][0]["text"]
    assert new_dir.exists()


def test_validate_jsonl_tool(tmp_path: pathlib.Path) -> None:
    """validate_jsonl 工具应能验证 JSONL 文件。"""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(
        '{"query": "q1", "answer": "a1"}\n{"query": "q2", "answer": "a2"}',
        encoding="utf-8",
    )

    from lazydspy.tools.data_ops import validate_jsonl

    result = run_async(validate_jsonl({
        "path": str(jsonl_file),
        "required_fields": ["query", "answer"],
    }))

    assert "验证通过" in result["content"][0]["text"]


def test_validate_jsonl_missing_field(tmp_path: pathlib.Path) -> None:
    """validate_jsonl 工具应检测缺失字段。"""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"query": "q1"}', encoding="utf-8")

    from lazydspy.tools.data_ops import validate_jsonl

    result = run_async(validate_jsonl({
        "path": str(jsonl_file),
        "required_fields": ["query", "answer"],
    }))

    assert "缺少字段" in result["content"][0]["text"]


def test_sample_data_tool() -> None:
    """sample_data 工具应能生成样例数据。"""
    from lazydspy.tools.data_ops import sample_data

    result = run_async(sample_data({
        "input_fields": ["query", "context"],
        "output_fields": ["answer"],
        "num_samples": 2,
    }))

    text = result["content"][0]["text"]
    assert "2 条样例数据" in text
    assert "query" in text
    assert "answer" in text


def test_estimate_cost_tool() -> None:
    """estimate_cost 工具应返回成本估算。"""
    from lazydspy.tools.domain_ops import estimate_cost

    result = run_async(estimate_cost({
        "optimizer": "gepa",
        "mode": "quick",
        "dataset_size": 50,
    }))

    text = result["content"][0]["text"]
    assert "成本估算结果" in text
    assert "预估成本" in text


def test_list_optimizers_tool() -> None:
    """list_optimizers 工具应列出所有优化器。"""
    from lazydspy.tools.domain_ops import list_optimizers

    result = run_async(list_optimizers({}))

    text = result["content"][0]["text"]
    assert "GEPA" in text
    assert "MIPROv2" in text


def test_get_defaults_tool() -> None:
    """get_defaults 工具应返回默认配置。"""
    from lazydspy.tools.domain_ops import get_defaults

    result = run_async(get_defaults({
        "optimizer": "gepa",
        "mode": "quick",
    }))

    text = result["content"][0]["text"]
    assert "breadth" in text
    assert "depth" in text
