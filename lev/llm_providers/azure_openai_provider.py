import json
import logging
import os
import pickle
import time
from typing import Any, Optional, cast

logger = logging.getLogger(__name__)

try:
    import openai
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, TokenCachePersistenceOptions
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
except ImportError:
    openai = None
    ChatCompletionMessageParam = None
    ChatCompletionMessageToolCall = None
    DefaultAzureCredential = None
    InteractiveBrowserCredential = None
    TokenCachePersistenceOptions = None

from lev.core.llm_provider import BaseLlmProvider, ModelResponse, ToolCall


class AzureOpenAIProvider(BaseLlmProvider):
    """Azure OpenAI provider with Azure Default Credentials and tool calling support."""

    # Simple file-based token cache
    _token_cache_file = os.path.expanduser("~/.azure_openai_token_cache")
    _cached_token: dict[str, Any] = {}

    def __init__(
        self,
        azure_endpoint: str,
        api_version: str = "2024-02-01",
        default_model: str = "gpt-4o-mini",
        api_key: str | None = None,
        use_azure_credentials: bool = True,
        cache_name: str = "azure_openai_cache",
        allow_unencrypted_storage: bool = False,
    ):
        """
        Initialize Azure OpenAI provider.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
            default_model: Default model deployment name
            api_key: Azure OpenAI API key (optional if using Azure credentials)
            use_azure_credentials: Whether to use Azure Default Credentials for authentication
            cache_name: Name for the token cache (used for persistent storage)
            allow_unencrypted_storage: Whether to allow unencrypted token storage on disk
        """
        if openai is None:
            raise ImportError("openai package is required for AzureOpenAIProvider")
        if use_azure_credentials and DefaultAzureCredential is None:
            raise ImportError("azure-identity package is required for Azure Default Credentials")

        super().__init__("azure-openai", supports_tools=True, default_model=default_model)

        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.use_azure_credentials = use_azure_credentials
        self.cache_name = cache_name
        self.allow_unencrypted_storage = allow_unencrypted_storage

        # Initialize the client
        if use_azure_credentials and not api_key:
            # Use Azure Default Credentials with token caching
            if TokenCachePersistenceOptions is None:
                raise ImportError(
                    "azure-identity >= 1.15.0 is required for token caching. "
                    "Install with: pip install 'azure-identity>=1.15.0'"
                )

            logger.info(
                f"AzureOpenAIProvider: Initializing with Azure credentials, cache_name='{cache_name}', allow_unencrypted={allow_unencrypted_storage}"
            )

            # Get initial token to establish authentication once
            self._ensure_credential(cache_name, allow_unencrypted_storage)

            # Use the credential directly - let Azure SDK handle token caching and refresh
            self.client = openai.AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                azure_ad_token_provider=self._get_azure_ad_token,
            )
        else:
            logger.info("AzureOpenAIProvider: Initializing with API key authentication")
            # Use API key authentication
            if not api_key:
                raise ValueError("Either api_key must be provided or use_azure_credentials must be True")

            self.client = openai.AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key,
            )

    @classmethod
    def from_config(cls, **kwargs: Any) -> "AzureOpenAIProvider":
        """Create Azure OpenAI provider from configuration."""
        import os

        azure_endpoint = kwargs.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint required (set AZURE_OPENAI_ENDPOINT or pass azure_endpoint)")

        api_version = kwargs.get("api_version", os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))
        api_key = kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        use_azure_credentials = kwargs.get("use_azure_credentials", True)

        # If no API key is provided and use_azure_credentials is True, use Azure Default Credentials
        if not api_key and not use_azure_credentials:
            raise ValueError(
                "Either Azure OpenAI API key required (set AZURE_OPENAI_API_KEY or pass api_key) or enable use_azure_credentials"
            )

        return cls(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            default_model=kwargs.get("model", "gpt-4o"),
            api_key=api_key,
            use_azure_credentials=use_azure_credentials,
            cache_name=kwargs.get("cache_name", "azure_openai_cache"),
            allow_unencrypted_storage=kwargs.get("allow_unencrypted_storage", False),
        )

    def _load_cached_token(self) -> dict[str, Any] | None:
        """Load token from file cache."""
        try:
            if os.path.exists(self._token_cache_file):
                with open(self._token_cache_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load token cache: {e}")
        return None

    def _save_token_to_cache(self, token_data: dict[str, Any]) -> None:
        """Save token to file cache."""
        try:
            with open(self._token_cache_file, "wb") as f:
                pickle.dump(token_data, f)
            logger.info(f"Token cached to {self._token_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save token cache: {e}")

    def _ensure_credential(self, cache_name: str, allow_unencrypted_storage: bool) -> None:
        """Ensure we have a valid token, using cache if possible."""
        logger.info("Checking for cached token...")

        # Try to load from cache first
        cached_data = self._load_cached_token()
        current_time = time.time()

        if cached_data and cached_data.get("expires_at", 0) > current_time + 300:  # 5 min buffer
            logger.info("Using cached token from file")
            self._cached_token = cached_data
            return

        logger.info("No valid cached token found, authenticating...")

        # Need to authenticate
        cache_options = TokenCachePersistenceOptions(
            name=cache_name,
            allow_unencrypted_storage=allow_unencrypted_storage,
        )

        credential = InteractiveBrowserCredential(cache_persistence_options=cache_options)

        logger.info("Getting token from Azure credential...")
        token = credential.get_token("https://cognitiveservices.azure.com/.default")

        # Cache the token data
        token_data = {
            "access_token": token.token,
            "expires_at": token.expires_on,
        }
        self._cached_token = token_data
        self._save_token_to_cache(token_data)

        logger.info(f"Token obtained and cached, expires at: {token.expires_on}")

    def _get_azure_ad_token(self) -> str:
        """
        Get Azure AD token for OpenAI service.
        This method is called by the OpenAI client automatically when needed.
        """
        logger.info("_get_azure_ad_token called")

        # Check if our cached token is still valid
        current_time = time.time()
        if self._cached_token and self._cached_token.get("expires_at", 0) > current_time + 300:  # 5 min buffer
            logger.info("Returning cached token")
            return self._cached_token["access_token"]

        # Token expired or missing, re-authenticate
        logger.warning("Token expired or missing, need to re-authenticate")
        self._ensure_credential(self.cache_name, self.allow_unencrypted_storage)
        return self._cached_token["access_token"]

    @property
    def model(self) -> str:
        """Current default model name."""
        return self.default_model

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1,
        model: str | None = None,
    ) -> ModelResponse:
        """Generate chat completion using Azure OpenAI API."""
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

            openai_messages.append(openai_msg)

        model_name = model or self.default_model

        # Prepare kwargs
        kwargs = {
            "model": model_name,
            "messages": openai_messages,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Add tools if provided
        if tools:
            # Tools should already be in OpenAI format from MCP client
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
            raise RuntimeError(f"Azure OpenAI API error: {str(e)}") from e
