from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    @classmethod
    def empty(cls) -> ModelResponse:
        return cls(content=None, tool_calls=[], finish_reason=None, usage=None)


class LlmProvider(Protocol):
    """Protocol for LLM providers supporting chat completion with optional tool calling."""

    @property
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @property
    def default_model(self) -> str | None:
        """Default model name if applicable."""
        ...

    @property
    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling."""
        ...

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        # TODO: tools should probably not be part of the base protocol
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse: ...


class BaseLlmProvider(ABC):
    """Base implementation providing common functionality."""

    def __init__(self, name: str, supports_tools: bool = False, default_model: str | None = None, **kwargs):
        self._name = name
        self._supports_tools = supports_tools
        self._default_model = default_model

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str | None:
        return self._default_model

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    def _validate_tool_support(self, tools: list[dict[str, Any]] | None) -> None:
        """Validate that tools are not requested if unsupported."""
        if tools and not self.supports_tools:
            raise ValueError(f"Provider {self.name} does not support tool calling")

    @abstractmethod
    async def chat_complete(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> ModelResponse:
        """Implementation must be provided by subclasses."""
        ...
