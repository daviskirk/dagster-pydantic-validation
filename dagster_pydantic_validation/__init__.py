from contextlib import suppress
from pathlib import Path
from typing import (
    Any,
    Collection,
    Mapping,
    Sequence,
    Type,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)
from uuid import UUID

import dagster
from dagster import get_dagster_logger
from pydantic import BaseModel

logger = get_dagster_logger()


@overload
def model_to_config_schema(model: Type[BaseModel]) -> dagster.Shape:
    ...


@overload
def model_to_config_schema(model: Any) -> Any:
    ...


def model_to_config_schema(model):
    """Convert pydantic model types to dagster config schema.

    >>> class Child(BaseModel):
    ...     a: int
    ...     b: str
    ...     c: tuple[int, ...] = ()
    ...
    >>> class Parent(BaseModel):
    ...     children: list[Child]
    ...     optional_child: Child | None = None
    ...
    >>> config_schema = model_to_config_schema(Parent)
    >>> @dagster.op(config_schema=config_schema)
    ... def test_op(context):
    ...     return context.op_config
    ...

    >>> context = dagster.build_op_context(config={"children": [{"a": 1, "b": "test", "c": [1, 2]}]})
    >>> result = test_op(context)
    >>> result
    {'children': [{'a': 1, 'b': 'test', 'c': [1, 2]}]}

    The config should now also be parsable by the pydantic model:

    >>> Parent.parse_obj(result)
    Parent(children=[Child(a=1, b='test', c=(1, 2))], optional_child=None)

    The same with work with pydantic data classes as well:
    >>> import pydantic
    >>> @pydantic.dataclasses.dataclass
    ... class D:
    ...     a: str
    ...     b: int | None = None
    ...
    >>> config_schema = model_to_config_schema(Parent)
    >>> context = dagster.build_op_context(config={"a": "test"})
    >>> result = test_op(context)

    Validation should work as expected:
    >>> context = dagster.build_op_context(config={"children": [{"a": 1, "b": "test", "c": ["wrong"]}]})
    >>> result = test_op(context)
    Traceback (most recent call last):
    ...
    dagster._core.errors.DagsterInvalidConfigError: Error in config for op
    Error 1: Invalid scalar at path root:config:children[0]:c[0]. Value "wrong" of type "<class 'str'>" is not valid for expected type "Int".

    """
    for type_annotation, dagster_type in {
        str: dagster.StringSource,
        int: dagster.IntSource,
        bool: dagster.BoolSource,
        Path: dagster.StringSource,
        UUID: dagster.StringSource,
    }.items():
        if _annotation_issubclass(model, type_annotation):
            return dagster_type

    if pydantic_model := getattr(model, "__pydantic_model__", None):
        model = pydantic_model

    if _annotation_issubclass(model, BaseModel):
        config_schema = {}
        for k, v in model.__fields__.items():
            outer_type: Any = v.outer_type_

            field_schema: Any
            if _annotation_issubclass(
                outer_type, Sequence
            ) and not _annotation_issubclass(outer_type, str):
                field_schema = dagster.Array(model_to_config_schema(v.type_))
            elif _annotation_issubclass(outer_type, Mapping):
                field_schema = dagster.Permissive()
            else:
                field_schema = model_to_config_schema(v.type_)

            if v.allow_none:
                field_schema = dagster.Noneable(field_schema)

            field_kwargs: dict[str, Any] = {}
            if not v.required:
                field_kwargs["is_required"] = False
            if v.field_info.description:
                field_kwargs["description"] = v.field_info.description
            config_schema[k] = (
                dagster.Field(field_schema, **field_kwargs)
                if field_kwargs
                else field_schema
            )
        return dagster.Shape(config_schema)
    else:
        return model


def _simplify_default(value: Any) -> Any:
    if not isinstance(value, str) and isinstance(value, Collection):
        return [_simplify_default(v) for v in value]
    if isinstance(value, BaseModel):
        return value.dict()
    return value


T = TypeVar("T")


def _annotation_issubclass(t: Any, class_or_tuple: T) -> TypeGuard[T]:
    with suppress(TypeError, ValueError):
        return issubclass(t, cast(Any, class_or_tuple))
    try:
        origin_type = t.__origin__
    except AttributeError:
        return False
    return _annotation_issubclass(origin_type, class_or_tuple)