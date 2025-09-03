from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import Any


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
