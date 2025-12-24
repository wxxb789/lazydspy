"""Command-line interface for lazydspy."""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, Iterable, List, Sequence
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
    mode: str = Field(default="quick", description="quick 或 full 模式")
    subset_size: int | None = Field(default=None, description="quick 模式的子集大小")
    checkpoint_needed: bool = Field(default=False, description="是否需要 checkpoint")
    checkpoint_dir: Path = Field(default=Path("checkpoints"), description="Checkpoint 目录")
    checkpoint_interval: int = Field(default=2, description="Checkpoint 间隔（步数）")
    max_checkpoints: int = Field(default=3, description="最多保留多少个 checkpoint")
    resume: bool = Field(default=False, description="是否尝试从 checkpoint 恢复")

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
    Question("mode", "偏好 quick 还是 full? 回答 quick 或 full。", post_process=lambda x: x.strip().lower()),
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
    table.add_row("模式", config.mode)
    subset_display = str(config.subset_size) if config.subset_size else "auto"
    table.add_row("采样数量", subset_display)
    table.add_row("Checkpoint 目录", str(config.checkpoint_dir))
    table.add_row("Checkpoint 间隔", str(config.checkpoint_interval))
    table.add_row("Checkpoint 数量上限", str(config.max_checkpoints))
    table.add_row("需要 checkpoint", "Yes" if config.checkpoint_needed else "No")
    table.add_row("默认 resume", "Yes" if config.resume else "No")
    console.print(table)


def _render_files(config: GenerationConfig) -> Path:
    base_dir = Path("generated") / config.session_id
    base_dir.mkdir(parents=True, exist_ok=True)

    script_path = base_dir / "pipeline.py"
    metadata_path = base_dir / "metadata.json"
    readme_path = base_dir / "README.md"

    metadata: Dict[str, Any] = {
        "python": ">=3.12",
        "dependencies": [
            "dspy",
            "pydantic>=2",
            "typer",
            "rich",
            "openai",
            "claude-agent-sdk",
            "anthropic",
        ],
        "scenario": config.scenario,
        "generated_at": datetime.utcnow().isoformat(),
        "generator": "lazydspy CLI",
    }

    generation_payload = config.model_dump(mode="json")
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
            #!/usr/bin/env python
            \"\"\"Auto-generated DSPy pipeline for $scenario.\"\"\"

            from __future__ import annotations

            import json
            from datetime import datetime
            from pathlib import Path
            from typing import Any, Dict, Iterable, List, Literal, Sequence

            import dspy
            import typer
            from pydantic import BaseModel, Field, ValidationError
            from rich.console import Console

            console = Console()

            METADATA: Dict[str, Any] = $metadata_json
            GENERATION_CONFIG: Dict[str, Any] = $generation_json


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


            def _build_optimizer(algorithm: str, hyperparameters: Dict[str, Any]) -> Any:
                normalized = algorithm.lower()
                if normalized == "gepa":
                    cls = getattr(dspy, "GEvalPromptedAssembly", None)
                else:
                    cls = getattr(dspy, "MIPROv2", None)
                if cls is None:
                    raise RuntimeError(f"DSPy 未提供 {algorithm}")
                return cls(metric=None, prompt_model=dspy.OpenAI(model="gpt-4o"), **hyperparameters)


            def _chunk_dataset(examples: List[dspy.Example], size: int) -> List[List[dspy.Example]]:
                if size <= 0:
                    return [examples]
                return [examples[idx : idx + size] for idx in range(0, len(examples), size)]


            def _save_checkpoint(program: Any, checkpoint_dir: Path, step: int, max_keep: int) -> Path:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "step": step,
                    "prompt": getattr(program, "prompt", ""),
                    "meta": getattr(program, "meta", {}),
                    "saved_at": datetime.utcnow().isoformat(),
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


            def _resolve_prompt(config: Dict[str, Any], checkpoint: Dict[str, Any] | None) -> str:
                if checkpoint and checkpoint.get("prompt"):
                    return str(checkpoint["prompt"])
                seed = config.get("seed_prompt") or "<<your prompt>>"
                scenario = config.get("scenario", "DSPy pipeline")
                return f"{seed}\\n[场景] {scenario}"


            app = typer.Typer(add_completion=False)


            @app.command()
            def run(
                mode: Literal["quick", "full"] = typer.Option(
                    "$mode_default", "--mode", "-m", help="运行模式：quick 或 full。"
                ),
                config: Path = typer.Option(
                    Path("metadata.json"), "--config", help="自定义配置路径，默认读取生成的 metadata.json。"
                ),
                checkpoint_dir: Path = typer.Option(
                    Path("$checkpoint_dir"), "--checkpoint-dir", help="Checkpoint 保存目录。"
                ),
                checkpoint_interval: int = typer.Option(
                    $checkpoint_interval, "--checkpoint-interval", min=1, help="每多少步写一次 checkpoint。"
                ),
                max_checkpoints: int = typer.Option(
                    $max_checkpoints, "--max-checkpoints", min=1, help="最多保留的 checkpoint 数量。"
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

                data_path = Path(runtime.get("data_path") or "data.jsonl")
                dev_path = data_path.with_name(data_path.stem + ".dev.jsonl")

                train_rows = _load_dataset(data_path)
                dev_rows = _load_dataset(dev_path) if dev_path.exists() else train_rows

                train_examples = _to_examples(train_rows, inputs)
                dev_examples = _to_examples(dev_rows, inputs)
                if not train_examples:
                    raise RuntimeError("缺少训练数据")

                effective_subset = subset_size if mode == "quick" and subset_size else len(train_examples)
                if mode == "quick" and effective_subset and effective_subset < len(train_examples):
                    train_examples = train_examples[:effective_subset]
                    dev_examples = dev_examples[: max(1, min(len(dev_examples), effective_subset))]

                optimizer = _build_optimizer(runtime["algorithm"], runtime.get("hyperparameters", {}))

                resume_state = _load_latest_checkpoint(checkpoint_dir) if resume else None
                seed_prompt = _resolve_prompt(runtime, resume_state)
                start_step = int(resume_state["step"]) + 1 if resume_state else 1

                batches = _chunk_dataset(
                    train_examples, effective_subset if mode == "quick" else len(train_examples)
                )
                if not batches:
                    raise RuntimeError("缺少训练数据")

                program = None
                for offset, batch in enumerate(batches, start=start_step):
                    program = optimizer.compile(trainset=batch, valset=dev_examples, seed_prompt=seed_prompt)
                    if offset % checkpoint_interval == 0:
                        saved = _save_checkpoint(program, checkpoint_dir, offset, max_checkpoints)
                        console.print(f"[green]已保存 checkpoint: {saved}[/]")

                if program is None:
                    raise RuntimeError("未能生成优化结果")

                final_prompt = getattr(program, "prompt", "")
                final_prompt_path = checkpoint_dir / "final_prompt.txt"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                final_prompt_path.write_text(str(final_prompt), encoding="utf-8")

                console.print("[bold cyan]最终 Prompt:[/]\\n" + str(final_prompt))
                console.print(f"[green]已保存最终 Prompt 到 {final_prompt_path}[/]")


            if __name__ == "__main__":
                app()
            """
        )
    )

    script = script_template.substitute(
        scenario=config.scenario,
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
    metadata_path.write_text(json.dumps(generation_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_content = textwrap.dedent(
        f"""\
        # Session {config.session_id}

        - 场景: {config.scenario}
        - 输入字段: {", ".join(config.input_fields)}
        - 输出字段: {", ".join(config.output_fields)}
        - 模式: {config.mode}
        - 模型偏好: {config.model_preference}
        - 算法: {config.algorithm}
        - 需要 checkpoint: {"是" if config.checkpoint_needed else "否"}
        - Checkpoint 间隔: {config.checkpoint_interval}, 上限: {config.max_checkpoints}

        运行示例：

        ```bash
        uv run {script_path} --mode {config.mode} --checkpoint-dir {config.checkpoint_dir}
        ```
        """
    )
    readme_path.write_text(readme_content, encoding="utf-8")

    return script_path


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
    _run_quality_checks(script_path)
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
