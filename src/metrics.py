"""模型评分指标，使用 LLM 生成并解析细则。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - 运行时环境可能缺少依赖
    OpenAI = None

from lazydspy.schemas import ScoreDetail


class _ChoiceMessage(TypedDict):
    """限定 OpenAI 返回的消息内容结构。"""

    content: str | None


class _Choice(TypedDict):
    """限定 OpenAI choices 字段结构。"""

    message: _ChoiceMessage


class _Completion(TypedDict):
    """限定 OpenAI 响应结构，用于类型检查。"""

    choices: List[_Choice]


# 评审提示，指导模型输出符合 ScoreDetail 约束的 JSON（含 feedback）。
JUDGE_PROMPT: str = (
    "你是严格的中文评审，请根据提供的参考答案与模型输出进行打分。"
    "必须返回 JSON 字符串，字段包括：\n"
    "- score: float，总分范围 0-5。\n"
    "- details: object，包含 relevance、accuracy、completeness、\n"
    "  clarity、formatting，均为 1-5 的整数。\n"
    "- feedback: str，对模型输出的中文反馈。\n"
    "只返回符合上述结构的 JSON，不要添加多余文字。"
)


def _safe_normalize_score(raw_score: float) -> float:
    """将任意数值裁剪并归一化到 0-1 区间。"""

    bounded = max(0.0, min(5.0, raw_score))
    return bounded / 5.0


def llm_judge_metric(gold: str, pred: str, trace: Any | None = None) -> float:
    """
    使用 GPT-4.1 评审模型输出，解析 JSON 到 ScoreDetail，返回归一化平均得分。

    当任意阶段（LLM 调用、JSON 解析、Pydantic 校验）发生异常时，回退返回 0.0。
    """

    try:
        if OpenAI is None:
            raise RuntimeError("openai 客户端不可用")

        # 构造 OpenAI 客户端并发送打分请求（trace 仅为兼容接口占位）。
        client: Any = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": f"参考答案：{gold}\n\n模型输出：{pred}",
                },
            ],
        )

        # 通过 TypedDict 限定最小访问属性，避免属性缺失导致的类型告警。
        raw: _Completion = {
            "choices": [
                {"message": {"content": completion.choices[0].message.content}}
            ]
        }

        # 提取 LLM 返回的 JSON 字符串并解析。
        content = raw["choices"][0]["message"]["content"] or ""
        payload: Dict[str, Any] = json.loads(content)

        # Pydantic 校验细节得分，确保范围正确。
        detail = ScoreDetail(**payload.get("details", {}))

        # 计算细则平均分与整体分，并归一化到 0-1。
        detail_values = [
            detail.relevance,
            detail.accuracy,
            detail.completeness,
            detail.clarity,
            detail.formatting,
        ]
        detail_avg = sum(detail_values) / len(detail_values)
        normalized_detail = _safe_normalize_score(detail_avg)

        overall = float(payload.get("score", detail_avg))
        normalized_overall = _safe_normalize_score(overall)

        # 综合整体分与细则平均分，取均值作为最终得分。
        final_score = (normalized_detail + normalized_overall) / 2
        return float(final_score)
    except Exception:
        # 任意异常均回退到 0.0，避免传播错误中断评测流程。
        return 0.0


__all__ = ["JUDGE_PROMPT", "llm_judge_metric"]
