"""轻量级 pydantic 兼容层，满足测试所需的基本校验功能。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar


class ConfigDict(dict[str, Any]):
    """轻量替代，用于兼容 pydantic.ConfigDict 行为."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


@dataclass
class FieldInfo:
    """字段元信息，记录默认值与范围约束。"""

    default: Any
    default_factory: Optional[Any] = None
    ge: Optional[float] = None
    le: Optional[float] = None
    description: Optional[str] = None


def Field(
    default: Any = Ellipsis,
    *,
    default_factory: Optional[Any] = None,
    ge: float | None = None,
    le: float | None = None,
    description: str | None = None,
) -> Any:
    """创建字段描述对象，用于约束范围。"""

    return FieldInfo(
        default=default,
        default_factory=default_factory,
        ge=ge,
        le=le,
        description=description,
    )


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

    # 模拟 pydantic 的 model_fields 暴露注解。
    model_fields: Dict[str, Any] = {}
    _model_validator_flag = "_is_model_validator"

    def __init__(self, **data: Any) -> None:
        errors: List[Dict[str, Any]] = []
        annotations = getattr(self.__class__, "__annotations__", {})
        # 将注解暴露为 model_fields，方便校验器读取。
        self.__class__.model_fields = dict(annotations)
        for name, annotation in annotations.items():
            field_obj = getattr(self.__class__, name, None)
            required = (
                isinstance(field_obj, FieldInfo)
                and field_obj.default is Ellipsis
                and field_obj.default_factory is None
            )
            if isinstance(field_obj, FieldInfo):
                if field_obj.default_factory is not None:
                    default = field_obj.default_factory()
                else:
                    default = field_obj.default
            else:
                default = field_obj

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

        # 执行 model_validator 标记的方法（after 模式）。
        for attribute in dir(self):
            candidate = getattr(self, attribute)
            if callable(candidate) and getattr(candidate, self._model_validator_flag, False):
                maybe_self = candidate()
                if maybe_self is not None:
                    # 允许校验器返回更新后的 self。
                    self = maybe_self

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.model_fields = dict(getattr(cls, "__annotations__", {}))

    @classmethod
    def model_validate(cls: Type[ModelT], data: Dict[str, Any]) -> ModelT:
        """兼容 pydantic 的替代校验方法。"""

        return cls(**data)

    def model_dump(self, mode: str | None = None) -> Dict[str, Any]:
        """返回当前实例数据字典。"""

        return dict(getattr(self, "__dict__", {}))


FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def field_validator(*_fields: str, **_kwargs: Any) -> Callable[[FuncT], FuncT]:
    """占位 field_validator，直接返回原函数。"""

    def decorator(func: FuncT) -> FuncT:
        return func

    return decorator


def model_validator(*_fields: str, **_kwargs: Any) -> Callable[[FuncT], FuncT]:
    """占位 model_validator，直接返回原函数。"""

    def decorator(func: FuncT) -> FuncT:
        setattr(func, BaseModel._model_validator_flag, True)
        return func

    return decorator


__all__ = [
    "BaseModel",
    "ConfigDict",
    "Field",
    "ValidationError",
    "field_validator",
    "model_validator",
]
