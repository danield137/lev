# LLM Provider Abstraction Layer

This module provides a unified interface for working with different LLM providers, eliminating the need for Semantic Kernel and enabling easy switching between providers.

## Features

- **Unified Interface**: Single `LlmProvider` protocol for all providers
- **Tool Calling Support**: Standardized tool calling across compatible providers
- **Optional Dependencies**: Graceful handling of missing packages
- **Async Support**: Full async/await compatibility
- **Configuration Management**: Environment-based and programmatic configuration

## Supported Providers

| Provider | Tool Calling | Required Package | Notes |
|----------|-------------|------------------|-------|
| OpenAI | ✅ | `openai` | GPT models, function calling |
| Azure OpenAI | ✅ | `openai`, `azure-identity` | Azure-hosted GPT models, supports Azure AD auth |
| Anthropic | ✅ | `anthropic` | Claude models, tool use |
| Ollama | ❌ | None | Local models, cost-effective for judges |

## Quick Start

### Using Environment Variables

```python
from lev.llm_providers import create_provider

# OpenAI (requires OPENAI_API_KEY)
provider = create_provider("openai")

# Azure OpenAI with Azure Default Credentials (recommended)
# Set AZURE_OPENAI_ENDPOINT in environment
provider = create_provider("azure-openai")

# Azure OpenAI with API key
# Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in environment
provider = create_provider("azure-openai", use_azure_credentials=False)

# Anthropic (requires ANTHROPIC_API_KEY)
provider = create_provider("anthropic")

# Local Ollama
provider = create_provider("ollama")
```

### Direct Instantiation

```python
from lev.llm_providers import OpenAIProvider, AzureOpenAIProvider, AnthropicProvider

# OpenAI
provider = OpenAIProvider(api_key="your-api-key")

# Azure OpenAI with Azure Default Credentials (recommended)
provider = AzureOpenAIProvider(
    azure_endpoint="https://your-resource.openai.azure.com",
    default_model="gpt-4o-mini",  # Your deployment name
    use_azure_credentials=True
)

# Azure OpenAI with API key
provider = AzureOpenAIProvider(
    azure_endpoint="https://your-resource.openai.azure.com",
    api_key="your-api-key",
    default_model="gpt-4o-mini",  # Your deployment name
    use_azure_credentials=False
)

# Anthropic
provider = AnthropicProvider(api_key="your-api-key")
```

## Azure OpenAI Configuration

### Environment Variables

For Azure OpenAI, you can configure the following environment variables:

- `AZURE_OPENAI_ENDPOINT` (required): Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY` (optional): API key for authentication
- `AZURE_OPENAI_API_VERSION` (optional, default: "2024-02-01"): API version to use
- `LLM_PROVIDER=azure-openai`: Set this to use Azure OpenAI as default provider

### Authentication Methods

#### 1. Azure Default Credentials with Token Caching (Recommended)

This method uses Azure's InteractiveBrowserCredential with persistent token caching to avoid repeated logins. The token is cached locally and reused across sessions:

```python
# Using factory with Azure Default Credentials and token caching
provider = create_provider("azure-openai", use_azure_credentials=True)

# Direct instantiation with custom cache settings
provider = AzureOpenAIProvider(
    azure_endpoint="https://your-resource.openai.azure.com",
    use_azure_credentials=True,
    cache_name="my_azure_openai_cache",  # Custom cache name
    allow_unencrypted_storage=True,      # Allow unencrypted cache on disk
)
```

**Token Cache Features:**
- Persistent storage across application restarts
- Interactive browser login required only on first use or token expiration
- Configurable cache name for isolation between applications
- Optional unencrypted storage (for development environments)
- Automatic token refresh when needed

**Requirements:**
- `azure-identity >= 1.15.0` for token caching support
- First-time authentication requires browser access

#### 2. API Key Authentication

```python
# Using factory with API key
provider = create_provider("azure-openai", 
                         api_key="your-api-key", 
                         use_azure_credentials=False)

# Direct instantiation
provider = AzureOpenAIProvider(
    azure_endpoint="https://your-resource.openai.azure.com",
    api_key="your-api-key",
    use_azure_credentials=False
)
```

### Setting up Azure Default Credentials

#### For Local Development:
1. Install Azure CLI: `az login`
2. Or set environment variables:
   ```bash
   export AZURE_CLIENT_ID="your-client-id"
   export AZURE_CLIENT_SECRET="your-client-secret"
   export AZURE_TENANT_ID="your-tenant-id"
   ```

#### For Production (Azure):
- Use Managed Identity (recommended)
- Or set the environment variables in your Azure App Service/Container Instance

## Usage Examples

### Basic Chat Completion

```python
import asyncio
from lev.llm_providers import create_provider

async def main():
    provider = create_provider("azure-openai")
    
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    response = await provider.chat_complete(messages)
    print(response.content)

asyncio.run(main())
```

### Tool Calling (Function Calling)

```python
import asyncio
from lev.llm_providers import create_tool_enabled_provider

async def main():
    provider = create_tool_enabled_provider("azure-openai")
    
    messages = [
        {"role": "user", "content": "What's the weather like in New York?"}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    response = await provider.chat_complete(messages, tools=tools)
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool: {tool_call.name}")
            print(f"Arguments: {tool_call.arguments}")

asyncio.run(main())
```

## Environment Setup

Create a `.env` file in your project root:

```env
# Choose your provider
LLM_PROVIDER=azure-openai

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-01

# Optional: Use API key instead of Azure Default Credentials
# AZURE_OPENAI_API_KEY=your-api-key

# Optional: Azure service principal (for Azure Default Credentials)
# AZURE_CLIENT_ID=your-client-id
# AZURE_CLIENT_SECRET=your-client-secret
# AZURE_TENANT_ID=your-tenant-id

# Alternative providers
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-anthropic-key
```

## Error Handling

The Azure OpenAI provider includes automatic token refresh for Azure Default Credentials and comprehensive error handling:

```python
try:
    response = await provider.chat_complete(messages)
except ImportError as e:
    print(f"Missing required packages: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
```

## Best Practices

1. **Use Azure Default Credentials** in production for better security
2. **Set deployment names as model names** in Azure OpenAI (not the base model names)
3. **Handle token expiration** - the provider automatically refreshes tokens
4. **Use environment variables** for configuration to avoid hardcoding credentials
5. **Test authentication** before deploying to production
