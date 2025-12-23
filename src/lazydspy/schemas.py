"""数据结构定义，提供评分细节与指标结果模型。"""

from pydantic import BaseModel, Field


class ScoreDetail(BaseModel):
    """各项评分细则（每项 1-5 分）。"""

    relevance: int = Field(  # 相关度
        ..., ge=1, le=5, description="答案与问题的相关度（1-5 分）"
    )
    accuracy: int = Field(  # 准确性
        ..., ge=1, le=5, description="事实与逻辑的准确性（1-5 分)"
    )
    completeness: int = Field(  # 完整性
        ..., ge=1, le=5, description="信息覆盖与细节完整度（1-5 分)"
    )
    clarity: int = Field(  # 清晰度
        ..., ge=1, le=5, description="表述清晰度与条理性（1-5 分)"
    )
    formatting: int = Field(  # 格式
        ..., ge=1, le=5, description="格式与可读性（1-5 分)"
    )


class MetricResult(BaseModel):
    """单条指标结果，包含总分、细项评分与文字反馈。"""

    score: float = Field(..., ge=0.0, le=5.0, description="总评分数（0-5 分）")
    details: ScoreDetail = Field(..., description="细项评分")
    feedback: str = Field(..., description="文字反馈")


__all__ = ["ScoreDetail", "MetricResult"]
