"""Tests for lazydspy new architecture.

Tests:
- knowledge/ - Optimizers and cost models
- tools.py - MCP tool functions
- prompts.py - System prompt configuration
- agent.py - Agent configuration
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
from conftest import run_async

# Ensure src directory is in path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


# ============================================================================
# Knowledge Tests
# ============================================================================


def test_get_optimizer_info() -> None:
    """Should get optimizer info by name."""
    from lazydspy.knowledge import get_optimizer_info

    gepa = get_optimizer_info("gepa")
    assert gepa is not None
    assert gepa.name == "GEPA"
    assert "general" in gepa.recommended_for

    mipro = get_optimizer_info("miprov2")
    assert mipro is not None
    assert mipro.name == "MIPROv2"


def test_list_all_optimizers() -> None:
    """Should list all available optimizers."""
    from lazydspy.knowledge import list_all_optimizers

    optimizers = list_all_optimizers()
    assert len(optimizers) == 2

    names = [opt["key"] for opt in optimizers]
    assert "gepa" in names
    assert "miprov2" in names


def test_get_recommended_optimizer() -> None:
    """Should recommend optimizer based on scenario."""
    from lazydspy.knowledge import get_recommended_optimizer

    assert get_recommended_optimizer("summary") == "gepa"
    assert get_recommended_optimizer("qa") == "miprov2"
    assert get_recommended_optimizer("unknown") == "gepa"


def test_estimate_optimization_cost() -> None:
    """Should estimate optimization cost."""
    from lazydspy.knowledge import estimate_optimization_cost

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
    """Full mode cost should be higher than quick mode."""
    from lazydspy.knowledge import estimate_optimization_cost

    quick = estimate_optimization_cost(optimizer="gepa", mode="quick", dataset_size=100)
    full = estimate_optimization_cost(optimizer="gepa", mode="full", dataset_size=100)

    assert full["estimated_cost_usd"] > quick["estimated_cost_usd"]


def test_list_supported_models() -> None:
    """Should list all supported models."""
    from lazydspy.knowledge import list_supported_models

    models = list_supported_models()
    assert "claude-opus-4.5" in models
    assert "gpt-4o" in models


# ============================================================================
# Tools Tests
# ============================================================================


def test_estimate_cost_tool() -> None:
    """estimate_cost tool should return cost estimation."""
    from lazydspy.tools import estimate_cost_impl

    result = run_async(
        estimate_cost_impl(
            {
                "optimizer": "gepa",
                "mode": "quick",
                "dataset_size": 50,
            }
        )
    )

    text = result["content"][0]["text"]
    data = json.loads(text)
    assert "estimated_cost_usd" in data
    assert "cost_hint" in data


def test_list_optimizers_tool() -> None:
    """list_optimizers tool should list all optimizers."""
    from lazydspy.tools import list_optimizers_impl

    result = run_async(list_optimizers_impl({}))

    text = result["content"][0]["text"]
    data = json.loads(text)
    assert "optimizers" in data
    assert len(data["optimizers"]) == 2


def test_get_defaults_tool() -> None:
    """get_defaults tool should return default configuration."""
    from lazydspy.tools import get_defaults_impl

    result = run_async(
        get_defaults_impl(
            {
                "optimizer": "gepa",
                "mode": "quick",
            }
        )
    )

    text = result["content"][0]["text"]
    data = json.loads(text)
    assert data["optimizer"] == "gepa"
    assert "hyperparameters" in data
    assert "breadth" in data["hyperparameters"]


def test_get_defaults_with_scenario() -> None:
    """get_defaults should recommend optimizer based on scenario."""
    from lazydspy.tools import get_defaults_impl

    result = run_async(
        get_defaults_impl(
            {
                "scenario": "qa",
                "mode": "quick",
            }
        )
    )

    text = result["content"][0]["text"]
    data = json.loads(text)
    assert data["optimizer"] == "miprov2"
    assert data["recommended_for_scenario"] == "qa"


def test_tool_names_format() -> None:
    """TOOL_NAMES should have correct MCP format."""
    from lazydspy.tools import TOOL_NAMES

    assert len(TOOL_NAMES) == 3
    for name in TOOL_NAMES:
        assert name.startswith("mcp__lazydspy__")


def test_create_mcp_server() -> None:
    """create_mcp_server should return server configuration."""
    from lazydspy.tools import create_mcp_server

    server = create_mcp_server()
    # The server is a TypedDict with 'name', 'type', and 'instance' keys
    assert server is not None
    assert server["name"] == "lazydspy"
    assert server["type"] == "sdk"
    assert "instance" in server


# ============================================================================
# Prompts Tests
# ============================================================================


def test_system_prompt_config() -> None:
    """get_system_prompt_config should return preset configuration."""
    from lazydspy.prompts import get_system_prompt_config

    config = get_system_prompt_config()
    assert config["type"] == "preset"
    assert config["preset"] == "claude_code"
    assert "append" in config
    assert "lazydspy" in config["append"]


def test_system_prompt_contains_key_info() -> None:
    """SYSTEM_PROMPT_APPEND should contain key instructions."""
    from lazydspy.prompts import SYSTEM_PROMPT_APPEND

    assert "DSPy" in SYSTEM_PROMPT_APPEND
    assert "优化器" in SYSTEM_PROMPT_APPEND or "optimizer" in SYSTEM_PROMPT_APPEND.lower()
    assert "mcp__lazydspy__" in SYSTEM_PROMPT_APPEND


# ============================================================================
# Agent Tests
# ============================================================================


def test_agent_config_defaults() -> None:
    """AgentConfig should have reasonable defaults."""
    from lazydspy.agent import AgentConfig

    config = AgentConfig()

    assert "claude" in config.model.lower() or "sonnet" in config.model.lower()
    assert config.base_url is None
    assert config.max_turns == 50
    assert config.debug is False


def test_agent_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """AgentConfig should read from environment variables."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
    monkeypatch.setenv("LAZYDSPY_DEBUG", "true")

    from lazydspy.agent import AgentConfig

    config = AgentConfig()

    assert config.model == "test-model"
    assert config.api_key == "test-key"
    assert config.debug is True


def test_agent_config_api_key_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """ANTHROPIC_AUTH_TOKEN should take priority over ANTHROPIC_API_KEY."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")

    from lazydspy.agent import AgentConfig

    config = AgentConfig()
    assert config.api_key == "auth-token"


def test_agent_config_fallback_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should fallback to ANTHROPIC_API_KEY when AUTH_TOKEN not set."""
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")

    from lazydspy.agent import AgentConfig

    config = AgentConfig()
    assert config.api_key == "api-key"


def test_agent_initialization() -> None:
    """Agent should initialize with config."""
    from lazydspy.agent import Agent, AgentConfig

    config = AgentConfig()
    agent = Agent(config)

    assert agent.config == config
    assert agent.console is not None


def test_agent_default_config() -> None:
    """Agent should use default config when none provided."""
    from lazydspy.agent import Agent

    agent = Agent()

    assert agent.config is not None
    assert agent.config.max_turns == 50
