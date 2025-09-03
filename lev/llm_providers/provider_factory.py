"""
Factory for creating LLM providers based on configuration.
"""

import os
from typing import Any

from dotenv import load_dotenv

from lev.core.llm_provider import LlmProvider
from lev.llm_providers import azure_openai_provider, lmstudio_provider, openai_provider

load_dotenv()


def create_provider(provider_name: str | None = None, **kwargs: Any) -> LlmProvider:
    """
    Create an LLM provider based on configuration.

    Args:
        provider_name: Provider name ("openai", "azure-openai", "lmstudio", "anthropic", "ollama")
                      If None, reads from LLM_PROVIDER env var
        **kwargs: Provider-specific configuration

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider name is unknown or required config missing
    """
    # Determine provider name
    if provider_name is None:
        provider_name = os.getenv("LLM_PROVIDER", "openai").lower()
    else:
        provider_name = provider_name.lower()

    # Create provider based on name using switch-case pattern
    if provider_name == "openai":
        return openai_provider.OpenAIProvider.from_config(**kwargs)
    elif provider_name == "azure_openai":
        return azure_openai_provider.AzureOpenAIProvider.from_config(**kwargs)
    elif provider_name == "lmstudio":
        return lmstudio_provider.LMStudioProvider.from_config(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. Supported: openai, azure_openai, lmstudio, anthropic, ollama"
        )


def create_tool_enabled_provider(provider_name: str | None = None, **kwargs: Any) -> LlmProvider:
    """
    Create a provider that supports tool calling.

    Args:
        provider_name: Provider name ("openai", "azure-openai", "lmstudio", "anthropic", "ollama")
                      If None, reads from LLM_PROVIDER env var
        **kwargs: Provider-specific configuration

    Raises:
        ValueError: If the configured provider doesn't support tools
    """
    provider = create_provider(provider_name, **kwargs)
    if not provider.supports_tools:
        raise ValueError(f"Provider {provider.name} does not support tool calling")
    return provider
