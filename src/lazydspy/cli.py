"""Command-line interface for lazydspy."""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List
from uuid import uuid4

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class GenerationConfig(BaseModel):
    """Validated configuration collected from the chat session."""

    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    scenario: str = Field(..., description="场景描述")
    input_fields: List[str] = Field(..., description="输入字段")
    output_fields: List[str] = Field(..., description="输出字段")
    model_preference: str = Field(..., description="模型偏好，例如 claude-3-5-sonnet-20241022")
    algorithm: str = Field(..., description="GEPA 或 MIPROv2")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="优化超参")
    data_path: Path | None = Field(default=None, description="数据路径")
    quick_mode: bool = Field(default=False, description="快速/完整偏好，True 表示 quick")
    checkpoint_needed: bool = Field(default=False, description="是否需要 checkpoint")

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


@dataclass
class Question:
    key: str
    hint: str
    post_process: Callable[[str], Any] | None = None


class ClaudeSession:
    """Lightweight wrapper to ask Claude for concise questions."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._client: Anthropic | None = None

    def _ensure_client(self) -> Anthropic | None:
        if self._client is not None:
            return self._client
        try:
            self._client = Anthropic()
        except Exception as exc:  # noqa: BLE001
            self._console.print(f"[yellow]无法初始化 Claude 客户端，将使用内置提示：{exc}[/]")
            self._client = None
        return self._client

    def ask(self, hint: str) -> str:
        client = self._ensure_client()
        if client is None:
            return hint

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=256,
                temperature=0.2,
                system=(
                    "你是 CLI 助手，请用一句简洁的中文问题向用户询问需要的配置字段，"
                    "保持礼貌且避免额外解释。"
                ),
                messages=[{"role": "user", "content": hint}],
            )
            content = response.content
            if not content:
                return hint
            first_block = content[0]
            text = getattr(first_block, "text", "")
            return text.strip() or hint
        except Exception as exc:  # noqa: BLE001
            self._console.print(f"[yellow]调用 Claude 失败，使用内置提示：{exc}[/]")
            return hint


QUESTIONS: List[Question] = [
    Question("scenario", "请针对 DSPy 任务描述应用场景，例如检索式问答/摘要等。"),
    Question("input_fields", "列出主要输入字段名称（逗号分隔），如 query, context。"),
    Question("output_fields", "列出输出字段名称（逗号分隔），如 answer, score。"),
    Question("model_preference", "偏好的模型名称或容量上限，例如 claude-3-5-sonnet。"),
    Question("algorithm", "选择优化算法（GEPA 或 MIPROv2），并说明原因。"),
    Question("hyperparameters", "提供关键超参，格式 key=value, key2=value2，例如 depth=3, lr=0.1。"),
    Question("data_path", "训练/验证数据路径，支持相对路径或绝对路径。"),
    Question("quick_mode", "偏好 quick 还是 full? 回答 quick 或 full。", post_process=lambda x: x.strip().lower() == "quick"),
    Question("checkpoint_needed", "是否需要中断续跑的 checkpoint? 回答 yes/no。", post_process=lambda x: x.strip().lower() in {"y", "yes", "true", "1"}),
]


def _ask_user(session: ClaudeSession, question: Question) -> Any:
    prompt = session.ask(question.hint)
    console.print(Panel(prompt, title="Claude 提问", expand=False))
    answer = input("> ").strip()
    return question.post_process(answer) if question.post_process else answer


def _summarize_config(config: GenerationConfig) -> None:
    table = Table(title="GenerationConfig 摘要", expand=False, show_lines=True)
    table.add_column("字段")
    table.add_column("值")
    table.add_row("Session", config.session_id)
    table.add_row("场景", config.scenario)
    table.add_row("输入字段", ", ".join(config.input_fields))
    table.add_row("输出字段", ", ".join(config.output_fields))
    table.add_row("模型偏好", config.model_preference)
    table.add_row("算法", config.algorithm)
    table.add_row("超参", json.dumps(config.hyperparameters, ensure_ascii=False))
    table.add_row("数据路径", str(config.data_path) if config.data_path else "<未指定>")
    table.add_row("偏好", "quick" if config.quick_mode else "full")
    table.add_row("需要 checkpoint", "Yes" if config.checkpoint_needed else "No")
    console.print(table)


def _render_files(config: GenerationConfig) -> Path:
    base_dir = Path("generated") / config.session_id
    base_dir.mkdir(parents=True, exist_ok=True)

    script_path = base_dir / "pipeline.py"
    metadata_path = base_dir / "metadata.json"
    readme_path = base_dir / "README.md"

    generation_payload = config.model_dump(mode="json")
    generation_payload["session_token"] = uuid4().hex

    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env python
        \"\"\"Auto-generated DSPy pipeline for {config.scenario}.\"\"\"

        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import Any, Dict, List

        import dspy
        from anthropic import Anthropic

        GENERATION_CONFIG: Dict[str, Any] = {json.dumps(generation_payload, ensure_ascii=False, indent=2)}


        def load_jsonl(path: Path) -> List[Dict[str, Any]]:
            data: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
            return data


        def build_lm(model: str) -> dspy.LM:
            return dspy.LM(model=model)


        def configure_models() -> None:
            lm = build_lm(GENERATION_CONFIG["model_preference"])
            teacher = dspy.OpenAI(model=GENERATION_CONFIG["model_preference"], temperature=0.7)
            dspy.configure(lm=lm, teacher_model=teacher, student_model=lm)


        def compile_program(trainset: List[dspy.Example], devset: List[dspy.Example]) -> Any:
            algorithm = GENERATION_CONFIG["algorithm"]
            hyper = GENERATION_CONFIG["hyperparameters"]
            if algorithm == "GEPA":
                Gepa = getattr(dspy, "GEvalPromptedAssembly", None)
                if Gepa is None:
                    raise RuntimeError("DSPy 未提供 GEvalPromptedAssembly")
                optimizer = Gepa(metric=None, prompt_model=dspy.OpenAI(model="gpt-4o"), **hyper)
            else:
                Mipro = getattr(dspy, "MIPROv2", None)
                if Mipro is None:
                    raise RuntimeError("DSPy 未提供 MIPROv2")
                optimizer = Mipro(metric=None, prompt_model=dspy.OpenAI(model="gpt-4o"), **hyper)
            return optimizer.compile(trainset=trainset, valset=devset, seed_prompt="<<your prompt>>")


        def build_examples(path: Path) -> List[dspy.Example]:
            raw = load_jsonl(path)
            inputs = GENERATION_CONFIG["input_fields"]
            return [dspy.Example(**item).with_inputs(*inputs) for item in raw]


        def main() -> None:
            data_path = Path(GENERATION_CONFIG["data_path"]) if GENERATION_CONFIG.get("data_path") else Path("data.jsonl")
            dev_path = data_path.with_name(data_path.stem + ".dev.jsonl")
            configure_models()
            trainset = build_examples(data_path)
            devset = build_examples(dev_path) if dev_path.exists() else trainset
            program = compile_program(trainset, devset)
            print("Compiled program:", getattr(program, "prompt", program))


        if __name__ == "__main__":
            main()
        """
    )

    script_path.write_text(script, encoding="utf-8")
    metadata_path.write_text(json.dumps(generation_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_content = textwrap.dedent(
        f"""\
        # Session {config.session_id}

        - 场景: {config.scenario}
        - 输入字段: {", ".join(config.input_fields)}
        - 输出字段: {", ".join(config.output_fields)}
        - 模型偏好: {config.model_preference}
        - 算法: {config.algorithm}
        - 快速/完整: {"quick" if config.quick_mode else "full"}
        - 需要 checkpoint: {"是" if config.checkpoint_needed else "否"}

        运行示例：

        ```bash
        uv run {script_path}
        ```
        """
    )
    readme_path.write_text(readme_content, encoding="utf-8")

    return script_path


def run_chat() -> None:
    console.print("[bold]开始 Claude 多轮问答，收集 GenerationConfig[/]")
    session = ClaudeSession(console)
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

    _summarize_config(config)
    confirm = input("确认生成脚本和附属文件? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        console.print("[yellow]已取消生成。[/]")
        return

    script_path = _render_files(config)
    console.print(f"[green]已生成脚本：{script_path}[/]")
    console.print(f"[green]示例运行命令：uv run {script_path}[/]")
    if config.checkpoint_needed:
        console.print("[blue]请在运行时确保 checkpoint 路径可用。[/]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="lazydspy CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="使用 Claude 多轮问答收集配置")
    chat_parser.set_defaults(func=lambda _args: run_chat())

    optimize_parser = subparsers.add_parser("optimize", help="运行 GEPA 优化示例")
    optimize_parser.set_defaults(func=lambda _args: run_optimize())

    return parser


def run_optimize() -> None:
    from lazydspy import optimize

    optimize.main()


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


__all__ = ["GenerationConfig", "main", "run_chat", "run_optimize"]
