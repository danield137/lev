import json
from typing import Any, Optional, cast

try:
    import openai
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
except ImportError:
    openai = None
    ChatCompletionMessageParam = None
    ChatCompletionMessageToolCall = None

from lev.core.llm_provider import BaseLlmProvider, ModelResponse, ToolCall


class LMStudioProvider(BaseLlmProvider):
    """LMStudio provider with OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        default_model: str = "gpt-oss",
        supports_tools: bool = True,
    ):
        if openai is None:
            raise ImportError("openai package is required for LMStudioProvider")

        super().__init__("lmstudio", supports_tools=supports_tools, default_model=default_model)

        # LMStudio runs a local OpenAI-compatible server
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,  # LMStudio doesn't require a real API key
        )

    @classmethod
    def from_config(cls, **kwargs: Any) -> "LMStudioProvider":
        """Create LMStudio provider from configuration."""
        import os

        base_url = kwargs.get("base_url") or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        api_key = kwargs.get("api_key") or os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        model = kwargs.get("model") or os.getenv("LMSTUDIO_MODEL", "gpt-oss")

        return cls(
            base_url=base_url,
            api_key=api_key,
            default_model=model,
            supports_tools=True,
        )

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1,
        model: str | None = None,
    ) -> ModelResponse:
        """Generate chat completion using LMStudio's OpenAI-compatible API."""
        self._validate_tool_support(tools)

        # Convert to OpenAI format, preserving all message fields
        openai_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            openai_msg = {"role": msg["role"]}  # type: ignore

            # Add content if present
            if "content" in msg:
                openai_msg["content"] = msg["content"]  # type: ignore

            # Add tool_calls if present (for assistant messages with tool calls)
            if "tool_calls" in msg:
                openai_msg["tool_calls"] = msg["tool_calls"]  # type: ignore

            # Add tool_call_id if present (for tool response messages)
            if "tool_call_id" in msg:
                openai_msg["tool_call_id"] = msg["tool_call_id"]  # type: ignore

            openai_messages.append(openai_msg) # type: ignore

        model_name = model or self.default_model

        # Prepare kwargs
        kwargs = {"model": model_name, "messages": openai_messages}

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if temperature is not None:
            kwargs["temperature"] = temperature

        # Add tools if provided and supported
        if tools and self.supports_tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # Extract tool calls if present
            tool_calls = None
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_call_typed = cast(ChatCompletionMessageToolCall, tc)
                    try:
                        arguments = json.loads(str(tool_call_typed.function.arguments))
                    except json.JSONDecodeError:
                        arguments = {}

                    tool_calls.append(
                        ToolCall(id=tool_call_typed.id, name=str(tool_call_typed.function.name), arguments=arguments)
                    )

            # Extract usage information
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return ModelResponse(
                content=cast(str, message.content) if message.content else None,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason,
                usage=usage,
            )

        except Exception as e:
            raise RuntimeError(f"LMStudio API error: {str(e)}") from e
