from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Mapping, Type, TypeVar, Union, cast, get_args, get_origin


def nested_get(data: Any, path: str, default: Any = None) -> Any:
    """
    Retrieve a nested value using dot-separated path from dict-like objects,
    dataclasses, or sequences of such objects.

    Args:
        data: Source object(s) - dict, dataclass, or sequence of such objects
        path: Dot-separated key/attribute path (e.g. "a.b.c")
        default: Value to return if path is not found

    Returns:
        Retrieved value or default if path not found

    Examples:
        nested_get({'a': {'b': 1}}, 'a.b')  # Returns 1
        nested_get([{'a': None}, {'a': {'b': 2}}], 'a.b', 0)  # Returns 2
    """

    def _get_single(obj: Any, path: str) -> tuple[bool, Any]:
        """Return (found, value) for single object traversal."""
        if obj is None:
            return False, None

        # Handle empty path - return the object itself
        if not path:
            return True, obj

        current = obj
        for segment in path.split("."):
            if isinstance(current, Mapping):
                if segment not in current:
                    return False, None
                current = current[segment]
            elif is_dataclass(current) and hasattr(current, segment):
                current = getattr(current, segment)
            elif hasattr(current, segment):
                current = getattr(current, segment)
            else:
                return False, None
        return True, current

    if data is None:
        return default

    # Handle sequences (list/tuple) with precedence
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        for item in data:
            found, value = _get_single(item, path)
            if found:
                return value
        return default

    # Handle single object
    found, value = _get_single(data, path)
    return value if found else default


T = TypeVar("T")


def from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
    """
    Generic, type-safe dataclass loader.
    Returns the exact type 'cls' (mypy/pyright infer T).
    Handles:
      - Nested dataclasses
      - Optional[T]
      - list[T] / tuple[T, ...] / dict[K, V] with dataclass values
      - Defaults and default_factory
      - Ignores unknown keys
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        key = f.name
        ftype = f.type

        if key not in data:
            if f.default is not MISSING:
                kwargs[key] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
                kwargs[key] = f.default_factory()  # type: ignore[attr-defined]
            else:
                kwargs[key] = None
            continue

        value = data[key]

        # Optional[T] or Union[T, None] (including Python 3.10+ | syntax)
        origin = get_origin(ftype)
        if (origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType)) and type(None) in get_args(
            ftype
        ):
            inner = next(t for t in get_args(ftype) if t is not type(None))
            if value is None:
                kwargs[key] = None
                continue
            ftype = inner

        # Nested dataclass
        if is_dataclass(ftype) and isinstance(value, Mapping):
            kwargs[key] = from_dict(ftype, value)  # type: ignore[arg-type]
            continue

        origin = get_origin(ftype)
        args = get_args(ftype)

        # list[T]
        if origin is list and args:
            elem_t = args[0]
            if is_dataclass(elem_t):
                kwargs[key] = [from_dict(elem_t, v) if isinstance(v, Mapping) else v for v in cast(list[Any], value)]  # type: ignore
            else:
                kwargs[key] = list(value)
            continue

        # tuple[T, ...]
        if origin is tuple and args:
            elem_t = args[0]
            if len(args) == 2 and args[1] is Ellipsis and is_dataclass(elem_t):
                kwargs[key] = tuple(from_dict(elem_t, v) if isinstance(v, Mapping) else v for v in cast(list[Any], value))  # type: ignore
            else:
                kwargs[key] = tuple(value)
            continue

        # dict[K, V]
        if origin is dict and len(args) == 2:
            _, val_t = args
            if is_dataclass(val_t):
                kwargs[key] = {
                    k: from_dict(val_t, v) if isinstance(v, Mapping) else v  # type: ignore
                    for k, v in cast(Mapping[Any, Any], value).items()
                }
            else:
                kwargs[key] = dict(value)
            continue

        kwargs[key] = value

    return cls(**kwargs)  # type: ignore[call-arg]
