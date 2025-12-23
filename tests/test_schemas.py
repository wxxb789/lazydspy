import pytest
from pydantic import ValidationError

from lazydspy.schemas import MetricResult, ScoreDetail


def test_score_detail_valid():
    detail = ScoreDetail(
        relevance=5,
        accuracy=4,
        completeness=3,
        clarity=2,
        formatting=1,
    )
    assert detail.relevance == 5
    assert detail.formatting == 1


def test_metric_result_valid():
    detail = ScoreDetail(
        relevance=5,
        accuracy=5,
        completeness=5,
        clarity=5,
        formatting=5,
    )
    result = MetricResult(score=4.5, details=detail, feedback="很好")
    assert result.score == 4.5
    assert result.details == detail
    assert result.feedback == "很好"


def test_score_detail_invalid_range():
    with pytest.raises(ValidationError) as excinfo:
        ScoreDetail(
            relevance=0,  # too low
            accuracy=6,  # too high
            completeness=3,
            clarity=3,
            formatting=3,
        )
    errors = excinfo.value.errors()
    assert any(
        err.get("loc") == ("relevance",) and err.get("ctx", {}).get("ge") == 1
        for err in errors
    )
    assert any(
        err.get("loc") == ("accuracy",) and err.get("ctx", {}).get("le") == 5
        for err in errors
    )


def test_metric_result_missing_fields():
    with pytest.raises(ValidationError) as excinfo:
        MetricResult()
    message = str(excinfo.value)
    assert "score" in message
    assert "details" in message
    assert "feedback" in message


def test_metric_result_score_out_of_bounds():
    with pytest.raises(ValidationError) as excinfo:
        MetricResult(
            score=5.5,
            details=ScoreDetail(
                relevance=3,
                accuracy=3,
                completeness=3,
                clarity=3,
                formatting=3,
            ),
            feedback="",
        )
    errors = excinfo.value.errors()
    assert any(
        err.get("loc") == ("score",) and err.get("ctx", {}).get("le") == 5.0
        for err in errors
    )
