from dataclasses import dataclass

from lev.common.extensions import nested_get


@dataclass
class SampleDataClass:
    a: dict
    b: str = "test"


@dataclass
class NestedSampleClass:
    inner: SampleDataClass
    value: int = 42


class TestNestedGet:
    def test_dict_simple(self):
        """Test simple dict access."""
        data = {"a": {"b": 1}}
        assert nested_get(data, "a.b") == 1
        assert nested_get(data, "a.c", "default") == "default"

    def test_dict_empty(self):
        """Test empty dict handling."""
        data = {}
        assert nested_get(data, "a.b", "default") == "default"

    def test_dict_none_value(self):
        """Test dict with None values."""
        data = {"a": None}
        assert nested_get(data, "a") is None
        assert nested_get(data, "a.b", "default") == "default"

    def test_dataclass_simple(self):
        """Test dataclass access."""
        data = SampleDataClass(a={"nested": "value"})
        assert nested_get(data, "b") == "test"
        assert nested_get(data, "a.nested") == "value"

    def test_dataclass_nested(self):
        """Test nested dataclass access."""
        inner = SampleDataClass(a={"deep": "nested"})
        data = NestedSampleClass(inner=inner)
        assert nested_get(data, "inner.b") == "test"
        assert nested_get(data, "inner.a.deep") == "nested"
        assert nested_get(data, "value") == 42

    def test_sequence_precedence(self):
        """Test sequence with precedence."""
        d1 = {"a": None}
        d2 = {"a": {"b": 1}}
        result = nested_get([d1, d2], "a.b", 2)
        assert result == 1

    def test_sequence_first_wins(self):
        """Test that first item in sequence takes precedence."""
        d1 = {"a": {"b": "first"}}
        d2 = {"a": {"b": "second"}}
        result = nested_get([d1, d2], "a.b")
        assert result == "first"

    def test_sequence_fallback(self):
        """Test sequence falls back to later items."""
        d1 = {"a": {"c": 1}}
        d2 = {"a": {"b": 2}}
        result = nested_get([d1, d2], "a.b", "default")
        assert result == 2

    def test_sequence_mixed_types(self):
        """Test sequence with mixed dict/dataclass."""
        d1 = {"a": None}
        d2 = SampleDataClass(a={"b": "found"})
        result = nested_get([d1, d2], "a.b", "default")
        assert result == "found"

    def test_sequence_all_miss(self):
        """Test sequence where no item has the path."""
        d1 = {"x": 1}
        d2 = {"y": 2}
        result = nested_get([d1, d2], "a.b", "default")
        assert result == "default"

    def test_none_input(self):
        """Test None input."""
        assert nested_get(None, "a.b", "default") == "default"

    def test_string_not_sequence(self):
        """Test that strings are not treated as sequences."""
        result = nested_get("test", "a.b", "default")
        assert result == "default"

    def test_object_with_attributes(self):
        """Test plain object with attributes."""

        class TestObj:
            def __init__(self):
                self.a = {"b": "value"}

        obj = TestObj()
        assert nested_get(obj, "a.b") == "value"

    def test_missing_attribute(self):
        """Test missing attribute on object."""

        class TestObj:
            pass

        obj = TestObj()
        assert nested_get(obj, "missing", "default") == "default"

    def test_empty_path(self):
        """Test empty path returns the object itself."""
        data = {"a": 1}
        assert nested_get(data, "", "default") == data

    def test_single_key(self):
        """Test single key access."""
        data = {"key": "value"}
        assert nested_get(data, "key") == "value"
        assert nested_get(data, "missing", "default") == "default"
