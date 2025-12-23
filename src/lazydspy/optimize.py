"""GEPA 优化流程示例，加载训练与验证数据并输出最佳提示。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import dspy

import core
import metrics

DATA_ROOT = Path(__file__).resolve().parent
DATA_DIR = DATA_ROOT / "data"


def _load_examples(path: Path) -> List[dspy.Example]:
    """从 JSONL 文件加载示例，保持与 dspy.Example 接口对齐。"""

    examples: List[dspy.Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(dspy.Example(**record).with_inputs("content"))
    return examples


def _build_examples() -> tuple[List[dspy.Example], List[dspy.Example]]:
    """读取训练与验证示例，各包含 5 条网页内容与预期摘要。"""

    train_path = DATA_DIR / "train.jsonl"
    dev_path = DATA_DIR / "dev.jsonl"

    return _load_examples(train_path), _load_examples(dev_path)


def main() -> None:
    """执行 GEPA 优化流程并输出最优提示。"""

    # 初始化 teacher/student 模型（占位实现，不实际请求远程服务）。
    teacher_lm = dspy.OpenAI(model="gpt-4o", temperature=0.7)
    student_lm = dspy.LM(model="gpt-4o-mini", temperature=0.7)

    # 配置 dspy，全局使用 student_lm 并保留 teacher_lm 以供评审。
    dspy.configure(lm=student_lm, teacher_model=teacher_lm, student_model=student_lm)

    # 构造训练与验证集。
    trainset, devset = _build_examples()

    # 初始化 GEPA 任务（使用占位实现时仅演示参数传递）。
    gepa = getattr(dspy, "GEvalPromptedAssembly", None)
    if gepa is None:
        raise RuntimeError("当前环境缺少 GEvalPromptedAssembly，占位实现无法继续。")

    optimizer = gepa(
        metric=metrics.llm_judge_metric,
        prompt_model=teacher_lm,
        breadth=5,
        depth=3,
        temperature=0.7,
    )

    # 记录初始提示并运行优化。
    print("\n===== 初始种子提示 =====")
    print(core.SEED_PROMPT)

    best_program = optimizer.compile(trainset=trainset, valset=devset, seed_prompt=core.SEED_PROMPT)

    print("\n===== 最优提示 =====")
    print(getattr(best_program, "prompt", "<无法获取提示>"))

    # 序列化最优状态，便于复用。
    state_path = DATA_ROOT / "optimized_state.json"
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(getattr(best_program, "__dict__", {}), f, ensure_ascii=False, indent=2)

    print(f"\n已将最优状态保存到 {state_path}")


if __name__ == "__main__":
    main()
