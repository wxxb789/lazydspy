"""轻量级 pydantic 兼容层，满足测试所需的基本校验功能。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar


@dataclass
class FieldInfo:
    """字段元信息，记录默认值与范围约束。"""

    default: Any
    ge: Optional[float] = None
    le: Optional[float] = None
    description: Optional[str] = None


def Field(
    default: Any,
    *,
    ge: float | None = None,
    le: float | None = None,
    description: str | None = None,
) -> Any:
    """创建字段描述对象，用于约束范围。"""

    return FieldInfo(default=default, ge=ge, le=le, description=description)


class ValidationError(Exception):
    """校验失败时抛出的异常，兼容 pydantic 接口。"""

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        super().__init__("; ".join(error.get("msg", "") for error in errors if "msg" in error))
        self._errors = errors

    def errors(self) -> List[Dict[str, Any]]:
        """返回结构化错误列表。"""

        return self._errors

    def __str__(self) -> str:
        fields = ", ".join("/".join(map(str, error.get("loc", ()))) for error in self._errors)
        return f"Validation error for: {fields}"


ModelT = TypeVar("ModelT", bound="BaseModel")


class BaseModel:
    """最小化的 BaseModel，实现必填与 ge/le 校验。"""

    def __init__(self, **data: Any) -> None:
        errors: List[Dict[str, Any]] = []
        annotations = getattr(self.__class__, "__annotations__", {})
        for name, annotation in annotations.items():
            field_obj = getattr(self.__class__, name, None)
            required = isinstance(field_obj, FieldInfo) and field_obj.default is Ellipsis
            default = field_obj.default if isinstance(field_obj, FieldInfo) else field_obj

            if name in data:
                value = data[name]
            elif required:
                errors.append({"loc": (name,), "msg": "Field required"})
                continue
            else:
                value = default

            if isinstance(field_obj, FieldInfo):
                if field_obj.ge is not None and value is not None and value < field_obj.ge:
                    errors.append(
                        {"loc": (name,), "msg": "Value must be >= ge", "ctx": {"ge": field_obj.ge}}
                    )
                if field_obj.le is not None and value is not None and value > field_obj.le:
                    errors.append(
                        {"loc": (name,), "msg": "Value must be <= le", "ctx": {"le": field_obj.le}}
                    )

            # 嵌套模型支持：若类型是 BaseModel 子类且传入字典，则递归构造。
            if (
                isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
                and isinstance(value, dict)
            ):
                value = annotation(**value)

            setattr(self, name, value)

        if errors:
            raise ValidationError(errors)

    @classmethod
    def model_validate(cls: Type[ModelT], data: Dict[str, Any]) -> ModelT:
        """兼容 pydantic 的替代校验方法。"""

        return cls(**data)


__all__ = ["BaseModel", "Field", "ValidationError"]
