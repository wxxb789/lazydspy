"""Command-line interface for lazydspy."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, cast
from uuid import uuid4

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lazydspy.models import (
    GEPA_PRESETS,
    MIPROV2_PRESETS,
    GEPAHyperparameters,
    MIPROv2Hyperparameters,
    RunMode,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

console = Console()


class GenerationConfig(BaseModel):
    """Validated configuration collected from the chat session."""

    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(default_factory=lambda: datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    scenario: str = Field(..., description="场景描述")
    input_fields: List[str] = Field(..., description="输入字段")
    output_fields: List[str] = Field(..., description="输出字段")
    model_preference: str = Field(..., description="模型偏好，例如 claude-3-5-sonnet-20241022")
    algorithm: str = Field(..., description="GEPA 或 MIPROv2")
    hyperparameters: GEPAHyperparameters | MIPROv2Hyperparameters | Dict[str, Any] = Field(
        default_factory=dict, description="优化超参（可为空，按模式应用默认值）"
    )
    data_path: Path | None = Field(default=None, description="数据路径")
    mode: str = Field(default="quick", description="quick 或 full 模式")
    subset_size: int | None = Field(default=None, description="quick 模式的子集大小")
    checkpoint_needed: bool = Field(default=False, description="是否需要 checkpoint")
    checkpoint_dir: Path = Field(default=Path("checkpoints"), description="Checkpoint 目录")
    checkpoint_interval: int = Field(default=1, description="Checkpoint 间隔（步数）")
    max_checkpoints: int = Field(default=20, description="最多保留多少个 checkpoint")
    resume: bool = Field(default=False, description="是否尝试从 checkpoint 恢复")
    generate_sample_data: bool = Field(
        default=False, description="是否生成 sample-data/train.jsonl 示例"
    )

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, value: str) -> str:
        normalized = value.lower().replace("-", "").replace("_", "")
        allowed = {"gepa", "miprov2", "mipro"}
        if normalized not in allowed:
            raise ValueError("algorithm must be GEPA or MIPROv2")
        return "GEPA" if normalized == "gepa" else "MIPROv2"

    @field_validator("input_fields", "output_fields", mode="before")
    @classmethod
    def parse_fields(cls, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        raise ValueError("fields must be a comma-separated string or list")

    @field_validator("hyperparameters", mode="before")
    @classmethod
    def parse_hyperparameters(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed: Dict[str, Any] = {}
            for chunk in value.split(","):
                if "=" not in chunk:
                    continue
                key, raw = chunk.split("=", maxsplit=1)
                key = key.strip()
                raw = raw.strip()
                if not key:
                    continue
                if raw.isdigit():
                    parsed[key] = int(raw)
                else:
                    try:
                        parsed[key] = float(raw)
                    except ValueError:
                        parsed[key] = raw
            return parsed
        raise ValueError("hyperparameters must be a mapping or key=value list")

    @field_validator("data_path", mode="before")
    @classmethod
    def parse_path(cls, value: Any) -> Path | None:
        if value is None or value == "":
            return None
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser()

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def parse_checkpoint_dir(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        if value is None or value == "":
            return Path("checkpoints")
        return Path(str(value)).expanduser()

    @field_validator("mode")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"quick", "full"}:
            raise ValueError("mode must be quick or full")
        return normalized

    @field_validator("subset_size")
    @classmethod
    def validate_subset_size(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("subset_size must be an integer") from exc
        if parsed <= 0:
            raise ValueError("subset_size must be positive")
        return parsed

    @field_validator("checkpoint_interval", "max_checkpoints")
    @classmethod
    def ensure_positive(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("must be a positive integer") from exc
        if parsed <= 0:
            raise ValueError("must be a positive integer")
        return parsed

    @model_validator(mode="after")
    def apply_hyperparameter_presets(self) -> "GenerationConfig":
        overrides: Dict[str, Any]
        if isinstance(self.hyperparameters, BaseModel):
            overrides = self.hyperparameters.model_dump()
        elif isinstance(self.hyperparameters, dict):
            overrides = {k: v for k, v in self.hyperparameters.items() if v is not None}
        else:
            overrides = {}

        normalized_algo = self.algorithm.lower().replace("-", "").replace("_", "")
        run_mode = cast(RunMode, self.mode)
        if normalized_algo == "gepa":
            self.hyperparameters = GEPAHyperparameters.from_mode(run_mode, overrides)
        else:
            self.hyperparameters = MIPROv2Hyperparameters.from_mode(run_mode, overrides)
        return self

    @property
    def active_hyperparameters(self) -> Dict[str, Any]:
        """Return the resolved hyperparameters dict for the selected algorithm."""
        if isinstance(self.hyperparameters, BaseModel):
            return self.hyperparameters.model_dump()
        if isinstance(self.hyperparameters, dict):
            return {k: v for k, v in self.hyperparameters.items() if v is not None}
        return {}


@dataclass
class Question:
    key: str
    hint: str
    post_process: Callable[[str], Any] | None = None


DEFAULT_SYSTEM_PROMPT = (
    "你是成本敏感的 CLI 助手，遵循“先简单后复杂”的原则："
    "先以最短的问题获取关键信息，再逐步补充细节，并提示用户关注推理成本。"
)


def _detect_scenario_type(scenario: str) -> str:
    lowered = scenario.lower()
    if any(keyword in lowered for keyword in ("摘要", "summary", "summarize", "总结")):
        return "summary"
    if any(keyword in lowered for keyword in ("检索", "retrieval", "search", "问答", "qa")):
        return "retrieval"
    if any(keyword in lowered for keyword in ("评分", "打分", "score", "judge", "评级")):
        return "scoring"
    return "general"


def _recommend_strategy(config: GenerationConfig) -> tuple[str, str, Dict[str, Any], str]:
    scenario_type = _detect_scenario_type(config.scenario)
    recommended_mode: RunMode = "quick"
    if scenario_type == "retrieval":
        algorithm = "MIPROv2"
        hyper = dict(MIPROV2_PRESETS[recommended_mode])
        cost_hint = (
            "检索类任务优先用 quick + MIPROv2（search_size 较小），"
            "full 搜索会放大上下文成本。"
        )
    elif scenario_type == "summary":
        algorithm = "GEPA"
        hyper = dict(GEPA_PRESETS[recommended_mode])
        cost_hint = "摘要类建议轻量 GEPA，快速收敛，必要时再切换 full 模式提升质量但成本更高。"
    elif scenario_type == "scoring":
        algorithm = "MIPROv2"
        hyper = dict(MIPROV2_PRESETS[recommended_mode])
        cost_hint = "评分/判别可用 quick + MIPROv2 保持稳定，full 模式会增加搜索轮次与 token 开销。"
    else:
        algorithm = "GEPA"
        hyper = dict(GEPA_PRESETS[recommended_mode])
        cost_hint = "通用场景默认走 quick + GEPA，先低成本验证，再按需提升容量。"
    return scenario_type, algorithm, hyper, cost_hint


class AgentSession:
    """Lightweight wrapper to orchestrate asking, confirming, and summarizing."""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        console: Console,
        system_prompt: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        auth_token: str | None = None,
        agent_env: Mapping[str, str] | None = None,
        denied_tools: Sequence[str] | None = None,
    ) -> None:
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._console = console
        self._session: Any | None = None
        self._model = model or self.DEFAULT_MODEL
        self._base_url = base_url
        self._auth_token = auth_token
        self._agent_env = {str(k): str(v) for k, v in (agent_env or {}).items() if str(v).strip()}
        self._denied_tools = [tool for tool in (denied_tools or []) if str(tool).strip()]

    def _build_sdk_session(self) -> Any:
        from claude_agent_sdk import Session  # type: ignore[attr-defined]

        if self._base_url:
            os.environ.setdefault("ANTHROPIC_BASE_URL", self._base_url)
        if self._auth_token:
            os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", self._auth_token)
            os.environ.setdefault("ANTHROPIC_API_KEY", self._auth_token)
        os.environ.setdefault("ANTHROPIC_MODEL", self._model)
        for key, value in self._agent_env.items():
            os.environ.setdefault(key, value)

        kwargs: Dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "model": self._model,
        }

        init_params = set(inspect.signature(Session).parameters)
        if self._base_url and "base_url" in init_params:
            kwargs["base_url"] = self._base_url
        if self._auth_token:
            if "api_key" in init_params:
                kwargs["api_key"] = self._auth_token
            elif "auth_token" in init_params:
                kwargs["auth_token"] = self._auth_token
        if self._agent_env and "env" in init_params:
            kwargs["env"] = dict(self._agent_env)
        if self._denied_tools and {"disallowed_tools", "denied_tools"} & init_params:
            target_key = "disallowed_tools" if "disallowed_tools" in init_params else "denied_tools"
            kwargs[target_key] = list(self._denied_tools)

        return Session(**kwargs)

    def _ensure_session(self) -> Any | None:
        if self._session is not None:
            return self._session
        try:
            self._session = self._build_sdk_session()
        except Exception as exc:  # noqa: BLE001
            self._console.print(f"[yellow]无法初始化 Claude 客户端，将使用内置提示：{exc}[/]")
            self._session = None
        return self._session

    def _invoke_session(self, session: Any, user_content: str) -> Any:
        messages_client = getattr(session, "messages", None)
        if messages_client and hasattr(messages_client, "create"):
            return messages_client.create(
                model=self._model,
                max_tokens=256,
                temperature=0.2,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )

        for candidate in ("complete", "send", "chat", "run"):
            handler = cast(Callable[[str], Any] | None, getattr(session, candidate, None))
            if callable(handler):
                return handler(user_content)

        raise RuntimeError("Claude Agent SDK session 不支持的接口")

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, Mapping):
            if "text" in response:
                return str(response["text"]).strip()
            content = response.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, Sequence):
                for block in content:
                    text: str | None
                    if isinstance(block, Mapping):
                        text = block.get("text") if isinstance(block.get("text"), str) else None
                    else:
                        candidate_text = getattr(block, "text", None)
                        text = candidate_text if isinstance(candidate_text, str) else None
                    if text:
                        return text.strip()

        if hasattr(response, "text"):
            return str(response.text).strip()

        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Sequence):
            for block in content:
                content_text: str | None
                if isinstance(block, Mapping):
                    content_text = (
                        block.get("text") if isinstance(block.get("text"), str) else None
                    )
                else:
                    candidate_text = getattr(block, "text", None)
                    content_text = candidate_text if isinstance(candidate_text, str) else None
                if content_text:
                    return content_text.strip()

        return ""

    def _call_model(self, user_content: str, fallback: str) -> str:
        session = self._ensure_session()
        if session is None:
            return fallback

        try:
            response = self._invoke_session(session, user_content)
            text = self._extract_response_text(response)
            return text or fallback
        except Exception as exc:  # noqa: BLE001
            self._console.print(f"[yellow]调用 Claude 失败，使用内置提示：{exc}[/]")
            return fallback

    def ask(self, hint: str) -> str:
        """Generate a concise question for the given hint."""
        prompt = textwrap.dedent(
            f"""\
            请根据下面的提示，生成一句简短的中文提问，引导用户回答对应字段。
            先简单后复杂，避免冗长，并提醒用户注意成本。

            提示：{hint}
            """
        )
        return self._call_model(prompt, fallback=hint)

    def confirm(self, summary: str) -> str:
        """Generate a confirmation message based on the current summary."""
        prompt = textwrap.dedent(
            f"""\
            根据以下配置摘要，生成一句简明的确认语句，提示用户输入 y/n 继续。
            保持礼貌，并提醒仅在必要时进行高成本操作。

            摘要：
            {summary}
            """
        )
        return self._call_model(prompt, fallback="确认上述配置并生成脚本吗？(y/N)")

    def summarize(self, answers: Mapping[str, Any]) -> str:
        """Summarize collected answers to help the user confirm quickly."""
        prompt = textwrap.dedent(
            f"""\
            用尽量短的中文总结以下配置要点，列出 3-5 个关键项，遵循先简单后复杂。
            避免过长描述，提示用户关注成本。

            配置：{json.dumps(dict(answers), ensure_ascii=False)}
            """
        )
        fallback_lines = [f"{key}: {value or '<空>'}" for key, value in answers.items()]
        fallback = "；".join(fallback_lines)
        return self._call_model(prompt, fallback=fallback)


QUESTIONS: List[Question] = [
    Question("scenario", "请针对 DSPy 任务描述应用场景，例如检索式问答/摘要等。"),
    Question("input_fields", "列出主要输入字段名称（逗号分隔），如 query, context。"),
    Question("output_fields", "列出输出字段名称（逗号分隔），如 answer, score。"),
    Question("model_preference", "偏好的模型名称或容量上限，例如 claude-3-5-sonnet。"),
    Question("algorithm", "选择优化算法（GEPA 或 MIPROv2），并说明原因。"),
    Question(
        "hyperparameters",
        (
            "如需覆盖 GEPA(breadth/depth/temperature) 或 MIPROv2(search_size 等) 的默认超参，"
            "请以 key=value 输入，留空沿用 quick/full 预设。"
        ),
    ),
    Question("data_path", "训练/验证数据路径，支持相对路径或绝对路径。"),
    Question(
        "mode",
        "偏好 quick 还是 full? 回答 quick 或 full。",
        post_process=lambda x: x.strip().lower(),
    ),
    Question(
        "subset_size",
        "quick 模式下希望采样多少条数据进行验证？输入正整数或留空。",
        post_process=lambda x: x.strip(),
    ),
    Question("checkpoint_dir", "checkpoint 保存目录（留空使用默认 checkpoints）"),
    Question(
        "checkpoint_interval",
        "每多少个 step 保存一次 checkpoint？输入正整数或留空。",
        post_process=lambda x: x.strip(),
    ),
    Question(
        "max_checkpoints",
        "最多保留多少个 checkpoint？输入正整数或留空。",
        post_process=lambda x: x.strip(),
    ),
    Question(
        "checkpoint_needed",
        "是否需要中断续跑的 checkpoint? 回答 yes/no。",
        post_process=lambda x: x.strip().lower() in {"y", "yes", "true", "1"},
    ),
    Question(
        "resume",
        "需要在运行时默认启用 --resume 吗？回答 yes/no。",
        post_process=lambda x: x.strip().lower() in {"y", "yes", "true", "1"},
    ),
    Question(
        "generate_sample_data",
        "需要生成 sample-data/train.jsonl 模板吗？回答 yes/no。",
        post_process=lambda x: x.strip().lower() in {"y", "yes", "true", "1"},
    ),
]


def _load_agent_profile(profile_path: Path | None) -> tuple[Dict[str, str], List[str]]:
    """Load agent env/permission hints from a JSON profile."""

    if profile_path is None:
        return {}, []

    resolved_path = profile_path.expanduser()
    if not resolved_path.exists():
        console.print(f"[yellow]未找到 agent 配置文件：{resolved_path}，已忽略。[/]")
        return {}, []

    try:
        raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]解析 agent 配置失败：{exc}，已忽略该文件。[/]")
        return {}, []

    env_data: Dict[str, str] = {}
    permissions: List[str] = []

    if isinstance(raw, Mapping) and isinstance(raw.get("env"), Mapping):
        env_data = {
            str(key): str(value)
            for key, value in raw["env"].items()
            if key and value is not None and str(value).strip()
        }

    perms_raw = raw.get("permissions") if isinstance(raw, Mapping) else None
    if isinstance(perms_raw, Mapping) and isinstance(perms_raw.get("deny"), Sequence):
        permissions = [str(item) for item in perms_raw["deny"] if str(item).strip()]

    return env_data, permissions


def _parse_agent_env_pairs(pairs: Sequence[str]) -> Dict[str, str]:
    """Parse KEY=VALUE pairs into a dict, ignoring invalid entries."""

    parsed: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            console.print(f"[yellow]忽略无效的 agent env：{pair}（缺少 =）[/]")
            continue
        key, value = pair.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            console.print(f"[yellow]忽略无效的 agent env：{pair}（缺少 key 或 value）[/]")
            continue
        parsed[key] = value
    return parsed


def _ask_user(session: AgentSession, question: Question) -> Any:
    prompt = session.ask(question.hint)
    console.print(Panel(prompt, title="Agent 提问", expand=False))
    answer = input("> ").strip()
    return question.post_process(answer) if question.post_process else answer


def _summarize_config(
    config: GenerationConfig,
    recommendation: tuple[str, str, Dict[str, Any], str] | None = None,
) -> None:
    table = Table(title="GenerationConfig 摘要", expand=False, show_lines=True)
    table.add_column("字段")
    table.add_column("值")
    table.add_row("Session", config.session_id)
    table.add_row("场景", config.scenario)
    if recommendation:
        scenario_type, algo, hyper, cost_hint = recommendation
        table.add_row("场景类型", scenario_type)
        table.add_row("推荐算法", f"{algo}（quick 轻量，可覆盖）")
        table.add_row("推荐超参", json.dumps(hyper, ensure_ascii=False))
        table.add_row("成本提示", cost_hint)
    table.add_row("输入字段", ", ".join(config.input_fields))
    table.add_row("输出字段", ", ".join(config.output_fields))
    table.add_row("模型偏好", config.model_preference)
    table.add_row("算法", config.algorithm)
    table.add_row("超参", json.dumps(config.active_hyperparameters, ensure_ascii=False))
    table.add_row("数据路径", str(config.data_path) if config.data_path else "<未指定>")
    table.add_row("模式", config.mode)
    subset_display = str(config.subset_size) if config.subset_size else "auto"
    table.add_row("采样数量", subset_display)
    table.add_row("Checkpoint 目录", str(config.checkpoint_dir))
    table.add_row("Checkpoint 间隔", str(config.checkpoint_interval))
    table.add_row("Checkpoint 数量上限", str(config.max_checkpoints))
    table.add_row("需要 checkpoint", "Yes" if config.checkpoint_needed else "No")
    table.add_row("默认 resume", "Yes" if config.resume else "No")
    table.add_row("生成样例数据", "Yes" if config.generate_sample_data else "No")
    console.print(table)


def _make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return _make_jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {key: _make_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(item) for item in value]
    return value


@dataclass
class RenderResult:
    script_path: Path
    metadata_path: Path
    readme_path: Path
    data_guide_path: Path
    data_guide_highlights: List[str]
    sample_data_path: Path | None


def _render_files(config: GenerationConfig) -> RenderResult:
    scenario_type, recommended_algo, recommended_hyper, cost_hint = _recommend_strategy(config)

    base_dir = Path("generated") / config.session_id
    base_dir.mkdir(parents=True, exist_ok=True)

    script_path = base_dir / "pipeline.py"
    metadata_path = base_dir / "metadata.json"
    readme_path = base_dir / "README.md"
    data_guide_path = base_dir / "DATA_GUIDE.md"
    sample_data_path: Path | None = None

    script_dependencies = [
        "dspy",
        "pydantic>=2",
        "typer",
        "rich",
        "openai",
        "anthropic",
    ]
    python_requirement = ">=3.12"

    metadata: Dict[str, Any] = {
        "python": python_requirement,
        "dependencies": script_dependencies,
        "scenario": config.scenario,
        "scenario_type": scenario_type,
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "lazydspy CLI",
        "cost_hint": cost_hint,
    }

    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        generation_payload = cast(Dict[str, Any], model_dump(mode="json"))
    else:  # pragma: no cover - compatibility for pydantic v1 in test stubs
        generation_payload = dict(getattr(config, "__dict__", {}))
    generation_payload = _make_jsonable(generation_payload)
    generation_payload["hyperparameters"] = config.active_hyperparameters
    generation_payload["hyperparameters_resolved"] = config.active_hyperparameters
    generation_payload["hyperparameter_defaults"] = {
        "gepa": GEPA_PRESETS,
        "miprov2": MIPROV2_PRESETS,
    }
    generation_payload["recommended"] = {
        "algorithm": recommended_algo,
        "hyperparameters": recommended_hyper,
        "scenario_type": scenario_type,
        "cost_hint": cost_hint,
    }
    generation_payload["session_token"] = uuid4().hex
    generation_payload["metadata"] = metadata

    data_fields = config.input_fields + config.output_fields
    data_model_fields = (
        textwrap.indent(
            "\n".join(
                f"{field}: str = Field(..., description=\"{field}\")"
                for field in data_fields
            ),
            "    ",
        )
        if data_fields
        else textwrap.indent("payload: str = Field(..., description=\"payload\")", "    ")
    )

    script_template = Template(
        textwrap.dedent(
            """\
            # /// script
            # requires-python = "$python_requirement"
            # dependencies = [
            $dependencies_block
            # ]
            # ///
            #!/usr/bin/env python
            \"\"\"Auto-generated DSPy pipeline for $scenario.\"\"\"

            from __future__ import annotations

            import json
            import math
            from datetime import UTC, datetime
            from pathlib import Path
            from typing import Any, Callable, Dict, Iterable, List, Literal, Sequence, Set

            import dspy
            import typer
            from pydantic import BaseModel, Field, ValidationError
            from rich.console import Console

            console = Console()

            METADATA: Dict[str, Any] = $metadata_json
            GENERATION_CONFIG: Dict[str, Any] = $generation_json
            SCENARIO_TYPE: str = GENERATION_CONFIG.get("recommended", {}).get(
                "scenario_type",
                GENERATION_CONFIG.get("metadata", {}).get("scenario_type", "general"),
            )
            COST_HINT: str = GENERATION_CONFIG.get("recommended", {}).get(
                "cost_hint", GENERATION_CONFIG.get("metadata", {}).get("cost_hint", "")
            )
            HYPERPARAMETER_DEFAULTS: Dict[str, Dict[str, Dict[str, Any]]] = GENERATION_CONFIG.get(
                "hyperparameter_defaults",
                {
                    "gepa": {
                        "quick": {"breadth": 2, "depth": 2, "temperature": 0.3},
                        "full": {"breadth": 4, "depth": 4, "temperature": 0.7},
                    },
                    "miprov2": {
                        "quick": {"search_size": 8, "temperature": 0.3},
                        "full": {"search_size": 16, "temperature": 0.6},
                    },
                },
            )


            class DataRow(BaseModel):
            $data_model_fields


            def _load_generation_config(config_path: Path) -> Dict[str, Any]:
                if not config_path.exists():
                    return dict(GENERATION_CONFIG)
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                merged = dict(GENERATION_CONFIG)
                merged.update(payload)
                return merged


            def _load_dataset(path: Path) -> List[DataRow]:
                rows: List[DataRow] = []
                with path.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f, start=1):
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        try:
                            rows.append(DataRow(**payload))
                        except ValidationError as exc:
                            raise ValueError(f"{path}:{idx} 校验失败: {exc}") from exc
                return rows


            def _to_examples(rows: Iterable[DataRow], inputs: Sequence[str]) -> List[dspy.Example]:
                examples: List[dspy.Example] = []
                for row in rows:
                    payload: Dict[str, Any] = {}
                    for key in row.__annotations__:
                        payload[key] = getattr(row, key)
                    examples.append(dspy.Example(**payload).with_inputs(*inputs))
                return examples


            def _select_hyperparameters(
                algorithm: str, mode: str, overrides: Dict[str, Any] | None
            ) -> Dict[str, Any]:
                normalized = algorithm.lower()
                defaults = HYPERPARAMETER_DEFAULTS.get(normalized, {})
                preset = dict(defaults.get(mode, defaults.get("quick", {})))
                for key, value in (overrides or {}).items():
                    if key in preset:
                        preset[key] = value
                return preset


            def _resolve_models(algorithm: str, mode: str, preference: str) -> tuple[Any, Any]:
                preferred = preference or "gpt-4o"
                fast_model = "gpt-4o-mini"
                normalized = algorithm.lower()
                if normalized == "gepa":
                    prompt_name = preferred if mode == "full" else fast_model
                else:
                    prompt_name = fast_model if mode == "quick" else preferred
                prompt_model = dspy.OpenAI(model=prompt_name)
                teacher_model = dspy.OpenAI(model=preferred)
                return prompt_model, teacher_model


            def _normalize_text(text: Any) -> str:
                return str(text or "").strip().lower()


            def _tokenize(text: Any) -> Set[str]:
                return set(_normalize_text(text).split())


            def _build_metric(
                output_fields: Sequence[str],
                scenario_type: str,
            ) -> Callable[[Any, Any, Any | None], float]:
                target_field = output_fields[0] if output_fields else None

                def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
                    if not target_field:
                        return 0.0
                    expected = getattr(example, target_field, None)
                    if expected is None and isinstance(example, dict):
                        expected = example.get(target_field)
                    actual = getattr(pred, target_field, None)
                    if actual is None and isinstance(pred, dict):
                        actual = pred.get(target_field)
                    if expected is None or actual is None:
                        return 0.0

                    normalized_expected = _normalize_text(expected)
                    normalized_actual = _normalize_text(actual)
                    if scenario_type == "summary":
                        expected_tokens = _tokenize(normalized_expected)
                        actual_tokens = _tokenize(normalized_actual)
                        overlap = len(expected_tokens & actual_tokens)
                        return overlap / max(1, len(expected_tokens))
                    if scenario_type == "retrieval":
                        if normalized_expected and normalized_expected in normalized_actual:
                            return 1.0
                        expected_tokens = _tokenize(normalized_expected)
                        actual_tokens = _tokenize(normalized_actual)
                        return len(expected_tokens & actual_tokens) / max(1, len(expected_tokens))
                    if scenario_type == "scoring":
                        try:
                            expected_score = float(normalized_expected)
                            actual_score = float(normalized_actual)
                        except ValueError:
                            return 0.0
                        gap = abs(expected_score - actual_score)
                        return max(0.0, 1.0 - gap / max(1.0, abs(expected_score)))
                    return 1.0 if normalized_expected == normalized_actual else 0.0

                return _metric


            def _build_optimizer(
                algorithm: str,
                hyperparameters: Dict[str, Any],
                mode: str,
                metric: Any,
                prompt_model: Any,
                teacher_model: Any,
            ) -> Any:
                normalized = algorithm.lower()
                if normalized == "gepa":
                    cls = getattr(dspy, "GEvalPromptedAssembly", None)
                else:
                    cls = getattr(dspy, "MIPROv2", None)
                if cls is None:
                    raise RuntimeError(f"DSPy 未提供 {algorithm}")
                configure = getattr(dspy, "configure", None)
                if callable(configure):
                    configure(lm=prompt_model, teacher_model=teacher_model)
                return cls(metric=metric, prompt_model=prompt_model, **hyperparameters)


            def _chunk_dataset(examples: List[dspy.Example], size: int) -> List[List[dspy.Example]]:
                if size <= 0:
                    return [examples]
                return [examples[idx : idx + size] for idx in range(0, len(examples), size)]


            def _determine_chunk_size(total_examples: int, max_checkpoints: int) -> int:
                target = max(1, min(20, max_checkpoints))
                if max_checkpoints >= 10:
                    target = max(target, 10)
                if total_examples <= 0:
                    return 0
                return max(1, math.ceil(total_examples / target))


            def _save_checkpoint(
                program: Any,
                checkpoint_dir: Path,
                step: int,
                max_keep: int,
                seed_prompt: str,
                best_score: float | None,
            ) -> Path:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "step": step,
                    "prompt": getattr(program, "prompt", ""),
                    "meta": getattr(program, "meta", {}),
                    "seed_prompt": seed_prompt,
                    "best_score": best_score,
                    "saved_at": datetime.now(UTC).isoformat(),
                }
                path = checkpoint_dir / f"checkpoint-{step:04d}.json"
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                checkpoints = sorted(checkpoint_dir.glob("checkpoint-*.json"))
                excess = len(checkpoints) - max_keep
                for stale in checkpoints[: max(0, excess)]:
                    stale.unlink()
                return path


            def _load_latest_checkpoint(checkpoint_dir: Path) -> Dict[str, Any] | None:
                checkpoints = sorted(checkpoint_dir.glob("checkpoint-*.json"))
                if not checkpoints:
                    return None
                latest = checkpoints[-1]
                return json.loads(latest.read_text(encoding="utf-8"))


            def _resolve_prompt(
                config: Dict[str, Any], checkpoint: Dict[str, Any] | None, scenario_type: str
            ) -> str:
                if checkpoint:
                    for key in ("seed_prompt", "prompt"):
                        if checkpoint.get(key):
                            return str(checkpoint[key])
                seed = config.get("seed_prompt")
                if not seed:
                    seed_templates = {
                        "summary": "请根据提供的输入生成简短摘要，突出关键信息，保持事实一致。",
                        "retrieval": "你是检索问答助手，根据给定上下文回答问题，避免臆造。",
                        "scoring": "作为评估员，请对候选回答进行打分并返回数值或评级，附简短原因。",
                    }
                    seed = seed_templates.get(scenario_type, "<<your prompt>>")
                scenario = config.get("scenario", "DSPy pipeline")
                return f"{seed}\\n[场景] {scenario}"


            app = typer.Typer(add_completion=False)


            @app.command()
            def run(
                mode: Literal["quick", "full"] = typer.Option(
                    "$mode_default", "--mode", "-m", help="运行模式：quick 或 full。"
                ),
                config: Path = typer.Option(
                    Path("metadata.json"),
                    "--config",
                    help="自定义配置路径，默认读取生成的 metadata.json。",
                ),
                checkpoint_dir: Path = typer.Option(
                    Path("$checkpoint_dir"), "--checkpoint-dir", help="Checkpoint 保存目录。"
                ),
                checkpoint_interval: int = typer.Option(
                    $checkpoint_interval,
                    "--checkpoint-interval",
                    min=1,
                    help="每多少步写一次 checkpoint。",
                ),
                max_checkpoints: int = typer.Option(
                    $max_checkpoints,
                    "--max-checkpoints",
                    min=1,
                    help="最多保留的 checkpoint 数量。",
                ),
                resume: bool = typer.Option(
                    $resume_default,
                    "--resume",
                    help="启用后会尝试从最新 checkpoint 恢复 seed prompt。",
                ),
                subset_size: int | None = typer.Option(
                    $subset_default,
                    "--subset-size",
                    min=1,
                    help="quick 模式下使用的采样大小。",
                ),
            ) -> None:
                runtime = _load_generation_config(config)
                inputs = list(runtime.get("input_fields", []))
                if not inputs:
                    console.print("[red]input_fields 不能为空[/]")
                    raise typer.Exit(code=1)

                if COST_HINT:
                    console.print(f"[yellow]{COST_HINT}[/]")

                data_path = Path(runtime.get("data_path") or "data.jsonl")
                dev_path = data_path.with_name(data_path.stem + ".dev.jsonl")

                train_rows = _load_dataset(data_path)
                dev_rows = _load_dataset(dev_path) if dev_path.exists() else train_rows

                train_examples = _to_examples(train_rows, inputs)
                dev_examples = _to_examples(dev_rows, inputs)
                if not train_examples:
                    raise RuntimeError("缺少训练数据")

                output_fields = list(runtime.get("output_fields", []))
                effective_subset = (
                    subset_size if mode == "quick" and subset_size else len(train_examples)
                )
                if mode == "quick" and effective_subset and effective_subset < len(train_examples):
                    train_examples = train_examples[:effective_subset]
                    dev_examples = dev_examples[: max(1, min(len(dev_examples), effective_subset))]

                metric = _build_metric(output_fields, SCENARIO_TYPE)
                hyper_overrides = runtime.get("hyperparameters_resolved") or runtime.get(
                    "hyperparameters", {}
                )
                hyperparameters = _select_hyperparameters(
                    runtime["algorithm"], mode, hyper_overrides
                )
                prompt_model, teacher_model = _resolve_models(
                    runtime["algorithm"], mode, runtime.get("model_preference", "")
                )
                optimizer = _build_optimizer(
                    runtime["algorithm"], hyperparameters, mode, metric, prompt_model, teacher_model
                )

                resume_state = _load_latest_checkpoint(checkpoint_dir) if resume else None
                seed_prompt = _resolve_prompt(runtime, resume_state, SCENARIO_TYPE)
                start_step = int(resume_state["step"]) + 1 if resume_state else 1
                if resume_state:
                    recovered_score = resume_state.get("best_score")
                    resume_message = f"从 step {resume_state.get('step', 0)} 恢复"
                    if recovered_score is not None:
                        resume_message += f"，score={recovered_score}"
                    console.print(f"[cyan]{resume_message}[/]")

                chunk_size = _determine_chunk_size(len(train_examples), max_checkpoints)
                batches = _chunk_dataset(train_examples, chunk_size)
                if not batches:
                    raise RuntimeError("缺少训练数据")

                program = None
                best_overall: float | None = None
                for offset, batch in enumerate(batches, start=start_step):
                    program = optimizer.compile(
                        trainset=batch, valset=dev_examples, seed_prompt=seed_prompt
                    )
                    seed_prompt = getattr(program, "prompt", seed_prompt)
                    meta = getattr(program, "meta", {}) or {}
                    best_score = getattr(program, "score", None) or meta.get("score") or meta.get(
                        "best_score"
                    )
                    best_overall = best_score if best_score is not None else best_overall
                    if offset % checkpoint_interval == 0:
                        saved = _save_checkpoint(
                            program,
                            checkpoint_dir,
                            offset,
                            max_checkpoints,
                            seed_prompt,
                            best_score if best_score is not None else best_overall,
                        )
                        console.print(f"[green]已保存 checkpoint: {saved}[/]")

                if program is None:
                    raise RuntimeError("未能生成优化结果")

                final_prompt = getattr(program, "prompt", "")
                final_prompt_path = checkpoint_dir / "final_prompt.txt"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                final_prompt_path.write_text(str(final_prompt), encoding="utf-8")

                if best_overall is not None:
                    console.print(f"[cyan]最佳得分: {best_overall}[/]")
                console.print("[bold cyan]最终 Prompt:[/]\\n" + str(final_prompt))
                console.print(f"[green]已保存最终 Prompt 到 {final_prompt_path}[/]")


            if __name__ == "__main__":
                app()
            """
        )
    )

    script = script_template.substitute(
        scenario=config.scenario,
        python_requirement=python_requirement,
        dependencies_block="\n".join(f"#     \"{dep}\"," for dep in script_dependencies),
        metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2),
        generation_json=json.dumps(generation_payload, ensure_ascii=False, indent=2),
        data_model_fields=data_model_fields,
        mode_default=config.mode,
        checkpoint_dir=str(config.checkpoint_dir),
        checkpoint_interval=config.checkpoint_interval,
        max_checkpoints=config.max_checkpoints,
        resume_default="True" if config.resume else "False",
        subset_default=config.subset_size if config.subset_size is not None else "None",
    )

    script_path.write_text(script, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(generation_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme_content = textwrap.dedent(
        f"""\
        # Session {config.session_id}

        - 场景: {config.scenario}
        - 输入字段: {", ".join(config.input_fields)}
        - 输出字段: {", ".join(config.output_fields)}
        - 模式: {config.mode}
        - 模型偏好: {config.model_preference}
        - 算法: {config.algorithm}
        - 推荐（轻量）: {recommended_algo} / {json.dumps(recommended_hyper, ensure_ascii=False)}
        - 成本提示: {cost_hint}
        - 需要 checkpoint: {"是" if config.checkpoint_needed else "否"}
        - Checkpoint 间隔: {config.checkpoint_interval}, 上限: {config.max_checkpoints}

        运行示例（依赖由嵌入的 PEP 723 块声明）：

        ```bash
        uv run {script_path} --mode {config.mode} --checkpoint-dir {config.checkpoint_dir}
        ```
        """
    )
    readme_path.write_text(readme_content, encoding="utf-8")

    io_fields = config.input_fields + config.output_fields
    io_section = "\n".join(f"- `{field}`: 请提供清洗后的字符串" for field in io_fields)
    if not io_section:
        io_section = "- 自由格式 payload"
    expectations = [
        f"场景：{config.scenario}",
        (
            f"输入字段：{', '.join(config.input_fields)}"
            if config.input_fields
            else "输入字段：<未指定>"
        ),
        (
            f"输出字段：{', '.join(config.output_fields)}"
            if config.output_fields
            else "输出字段：<未指定>"
        ),
        f"数据路径：{config.data_path or '<未指定>'}",
    ]
    data_guide_lines = [
        "# 数据准备指引",
        "",
        f"- 目标场景：{config.scenario}",
        "- 建议：先用少量样例验证格式，再逐步扩充数据量，避免一次性投入高成本。",
        "- 格式：UTF-8 编码的 JSONL，每行一个独立样本。",
        "- 字段要求：",
        io_section,
        "",
        "## 最小可用样例",
        "```json",
    ]
    sample_row = {field: f"<填写{field}>" for field in io_fields}
    if not sample_row:
        sample_row = {"payload": "<<your content>>"}
    data_guide_lines.append(json.dumps(sample_row, ensure_ascii=False))
    data_guide_lines.append("```")
    data_guide_lines.append("")
    data_guide_lines.append("## 质量检查清单")
    data_guide_lines.extend(
        [
            "- 检查空值/缺字段并补全或剔除。",
            "- 确认每行 JSON 可独立解析，避免尾随逗号。",
            "- 若含标注字段（如 answer/label），确保无冲突或歧义。",
        ]
    )
    data_guide_path.write_text("\n".join(data_guide_lines), encoding="utf-8")

    if config.generate_sample_data:
        sample_dir = Path("sample-data")
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_data_path = sample_dir / "train.jsonl"
        example_rows = [
            {field: f"示例 {idx} 的 {field}" for field in io_fields} for idx in range(1, 3)
        ]
        if not example_rows or not example_rows[0]:
            example_rows = [{"payload": f"示例 {idx}"} for idx in range(1, 3)]
        sample_data_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in example_rows),
            encoding="utf-8",
        )

    data_guide_highlights = expectations + [
        f"推荐（轻量）：{recommended_algo} / {json.dumps(recommended_hyper, ensure_ascii=False)}",
        f"成本提示：{cost_hint}",
        "数据指引路径：" + str(data_guide_path),
    ]

    return RenderResult(
        script_path=script_path,
        metadata_path=metadata_path,
        readme_path=readme_path,
        data_guide_path=data_guide_path,
        data_guide_highlights=data_guide_highlights,
        sample_data_path=sample_data_path,
    )


def _run_command(command: Sequence[str]) -> bool:
    console.print(f"[blue]运行：{' '.join(command)}[/]")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        console.print(result.stdout)
    if result.stderr:
        console.print(result.stderr)
    return result.returncode == 0


def _run_quality_checks(script_path: Path) -> None:
    checks = [
        ("ruff", ["uv", "run", "ruff", "check", str(script_path)]),
        ("mypy", ["uv", "run", "mypy", str(script_path)]),
    ]
    for name, command in checks:
        success = _run_command(command)
        if success:
            console.print(f"[green]{name} 校验通过。[/]")
        else:
            console.print(f"[red]{name} 校验未通过，请根据输出修复 {script_path}。[/]")


def run_chat(
    *,
    model: str | None = None,
    base_url: str | None = None,
    auth_token: str | None = None,
    agent_config: Path | None = None,
    agent_env: Mapping[str, str] | None = None,
    deny_permissions: Sequence[str] | None = None,
) -> None:
    console.print("[bold]开始多轮问答，收集 GenerationConfig[/]")
    session = _build_agent_session(
        model=model,
        base_url=base_url,
        auth_token=auth_token,
        agent_config=agent_config,
        agent_env=agent_env,
        deny_permissions=deny_permissions,
    )
    answers: Dict[str, Any] = {}
    for question in QUESTIONS:
        answers[question.key] = _ask_user(session, question)

    try:
        config = GenerationConfig.model_validate(answers)
    except ValidationError as exc:
        console.print(f"[red]配置校验失败：{exc}[/]")
        for error in exc.errors():
            loc = " -> ".join(str(item) for item in error.get("loc", ()))
            msg = error.get("msg", "")
            console.print(f"[red]- {loc}: {msg}[/]")
        return

    recommendation = _recommend_strategy(config)
    _summarize_config(config, recommendation=recommendation)
    summary = session.summarize(
        {
            **answers,
            "recommended_algorithm": recommendation[1],
            "recommended_hyperparameters": recommendation[2],
        }
    )
    console.print(Panel(summary, title="AI 总结", expand=False))
    console.print(Panel(recommendation[3], title="成本提示（轻量起步，可覆盖）", expand=False))
    confirm_prompt = session.confirm(summary)
    console.print(Panel(confirm_prompt, title="确认", expand=False))
    confirm = input("> ").strip().lower()
    if confirm not in {"y", "yes"}:
        console.print("[yellow]已取消生成。[/]")
        return

    render_result = _render_files(config)
    console.print(f"[green]已生成脚本：{render_result.script_path}[/]")
    console.print(
        f"[green]示例运行命令（PEP 723 嵌入式依赖）：uv run {render_result.script_path}[/]"
    )

    guide_table = Table(title="数据指引要点", expand=False, show_header=False)
    for line in render_result.data_guide_highlights:
        guide_table.add_row(line)
    console.print(guide_table)

    if render_result.sample_data_path:
        console.print(f"[cyan]已生成样例数据：{render_result.sample_data_path}[/]")

    _run_quality_checks(render_result.script_path)
    if config.checkpoint_needed:
        console.print("[blue]请在运行时确保 checkpoint 路径可用。[/]")


def main(argv: Iterable[str] | None = None) -> None:
    args = list(argv) if argv is not None else None
    try:
        app(args=args, standalone_mode=False)
    except typer.Exit as exc:  # pragma: no cover - passthrough for Typer exit handling
        raise SystemExit(exc.exit_code) from exc


def _build_agent_session(
    *,
    model: str | None = None,
    base_url: str | None = None,
    auth_token: str | None = None,
    agent_config: Path | None = None,
    agent_env: Mapping[str, str] | None = None,
    deny_permissions: Sequence[str] | None = None,
) -> AgentSession:
    system_prompt = os.getenv("LAZYDSPY_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT

    profile_env, profile_denies = _load_agent_profile(agent_config)

    merged_env: Dict[str, str] = {}
    merged_env.update(profile_env)
    if agent_env:
        merged_env.update({str(k): str(v) for k, v in agent_env.items() if str(v).strip()})

    for key, value in merged_env.items():
        os.environ.setdefault(key, value)

    merged_denies: List[str] = list(profile_denies)
    if deny_permissions:
        merged_denies.extend([str(item) for item in deny_permissions if str(item).strip()])
    merged_denies = list(dict.fromkeys(merged_denies))  # 去重并保持顺序

    def _normalize(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    user_model = _normalize(model)
    profile_model = _normalize(merged_env.get("ANTHROPIC_MODEL"))
    env_model = _normalize(os.getenv("ANTHROPIC_MODEL"))
    resolved_model = user_model or profile_model or env_model or AgentSession.DEFAULT_MODEL

    user_base_url = _normalize(base_url)
    if base_url is not None and not user_base_url:
        console.print(
            "[yellow]传入的 --base-url 为空，已忽略并回退到默认 Anthropic Endpoint。[/]"
        )

    resolved_base_url = (
        user_base_url or _normalize(merged_env.get("ANTHROPIC_BASE_URL")) or _normalize(os.getenv("ANTHROPIC_BASE_URL"))
    )
    resolved_auth_token = (
        _normalize(auth_token)
        or _normalize(merged_env.get("ANTHROPIC_AUTH_TOKEN"))
        or _normalize(merged_env.get("ANTHROPIC_API_KEY"))
        or _normalize(os.getenv("ANTHROPIC_AUTH_TOKEN"))
        or _normalize(os.getenv("ANTHROPIC_API_KEY"))
    )

    if resolved_base_url and not resolved_auth_token:
        console.print(
            "[yellow]检测到 ANTHROPIC_BASE_URL 但未提供 ANTHROPIC_AUTH_TOKEN/ANTHROPIC_API_KEY，"
            "可能导致自定义 Claude Endpoint 初始化失败。[/]"
        )

    if model is not None and not user_model:
        console.print(
            "[yellow]未提供有效的模型名称，已回退到默认 claude-3-5-sonnet-20241022。[/]"
        )

    return AgentSession(
        console,
        system_prompt=system_prompt,
        model=resolved_model,
        base_url=resolved_base_url,
        auth_token=resolved_auth_token,
        agent_env=merged_env,
        denied_tools=merged_denies,
    )


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Rich/ Typer 驱动的 lazydspy CLI，通过 chat 子命令生成脚本，"
        "并使用生成的脚本执行优化。"
    ),
)


@app.command(help="使用 Claude Agent SDK 交互式收集 DSPy 配置。")
def chat(
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        envvar="ANTHROPIC_MODEL",
        help="自定义 Claude 模型名称（默认 claude-3-5-sonnet-20241022）。",
    ),
    base_url: str = typer.Option(
        None,
        "--base-url",
        envvar="ANTHROPIC_BASE_URL",
        help="自定义 Claude Endpoint，例如本地代理。",
    ),
    auth_token: str = typer.Option(
        None,
        "--auth-token",
        envvar=["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"],
        help="Claude API Token，支持本地代理或官方 Key。",
    ),
    agent_config: Path | None = typer.Option(
        None,
        "--agent-config",
        help="读取包含 env/permissions 字段的 JSON 配置文件，用于 Claude Agent SDK。",
    ),
    agent_env: List[str] | None = typer.Option(
        None,
        "--agent-env",
        "-E",
        help="以 KEY=VALUE 形式追加 Claude Agent SDK 环境变量，可重复传入。",
    ),
    deny_permission: List[str] | None = typer.Option(
        None,
        "--deny-permission",
        help="显式禁用的工具/权限名称，可重复传入；与 agent 配置文件合并。",
    ),
) -> None:
    run_chat(
        model=model,
        base_url=base_url,
        auth_token=auth_token,
        agent_config=agent_config,
        agent_env=_parse_agent_env_pairs(agent_env or []),
        deny_permissions=deny_permission or [],
    )


__all__ = ["AgentSession", "GenerationConfig", "app", "main", "run_chat"]
