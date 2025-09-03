"""
Tests for LMStudio provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lev.core.llm_provider import ModelResponse
from lev.llm_providers.lmstudio_provider import LMStudioProvider
from lev.llm_providers.provider_factory import create_provider


class TestLMStudioProvider:
    """Test LMStudio provider functionality."""

    def test_provider_initialization(self):
        """Test provider can be initialized with default values."""
        provider = LMStudioProvider()

        assert provider.name == "lmstudio"
        assert provider.default_model == "gpt-oss"
        assert provider.supports_tools is True

    def test_provider_initialization_with_custom_values(self):
        """Test provider can be initialized with custom values."""
        provider = LMStudioProvider(
            base_url="http://localhost:5000/v1",
            api_key="custom-key",
            default_model="custom-model",
            supports_tools=True,
        )

        assert provider.name == "lmstudio"
        assert provider.default_model == "custom-model"
        assert provider.supports_tools is True

    def test_from_config_with_defaults(self):
        """Test creating provider from config with default values."""
        provider = LMStudioProvider.from_config()

        assert provider.name == "lmstudio"
        assert provider.default_model == "gpt-oss"
        assert provider.supports_tools is True

    @patch.dict(
        "os.environ",
        {
            "LMSTUDIO_BASE_URL": "http://localhost:5000/v1",
            "LMSTUDIO_MODEL": "custom-model",
            "LMSTUDIO_SUPPORTS_TOOLS": "true",
        },
    )
    def test_from_config_with_env_vars(self):
        """Test creating provider from config with environment variables."""
        provider = LMStudioProvider.from_config()

        assert provider.name == "lmstudio"
        assert provider.default_model == "custom-model"
        assert provider.supports_tools is True

    def test_provider_factory_integration(self):
        """Test that the provider can be created through the factory."""
        provider = create_provider("lmstudio")

        assert provider.name == "lmstudio"
        assert isinstance(provider, LMStudioProvider)

    @pytest.mark.asyncio
    @patch("lev.llm_providers.lmstudio_provider.openai")
    async def test_chat_complete_basic(self, mock_openai):
        """Test basic chat completion functionality."""
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        provider = LMStudioProvider()
        messages = [{"role": "user", "content": "Hello"}]

        result = await provider.chat_complete(messages)

        assert isinstance(result, ModelResponse)
        assert result.content == "Hello, world!"
        assert result.tool_calls is None
        assert result.finish_reason == "stop"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_tool_validation_when_not_supported(self):
        """Test that tools are rejected when not supported."""
        provider = LMStudioProvider(supports_tools=False)
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]

        with pytest.raises(ValueError, match="does not support tool calling"):
            # This should be sync since _validate_tool_support is sync
            provider._validate_tool_support(tools)

    @patch("lev.llm_providers.lmstudio_provider.openai", None)
    def test_missing_openai_dependency(self):
        """Test that ImportError is raised when openai is not available."""
        with pytest.raises(ImportError, match="openai package is required"):
            LMStudioProvider()
