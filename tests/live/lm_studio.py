"""
Live test for LMStudio provider.

This test requires a running LMStudio instance with a model loaded.
Run LMStudio and start the local server before running this test.

Usage:
    python tests/live/lm_studio.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lev.llm_providers.provider_factory import create_provider


async def test_lmstudio_connection():
    """Test basic connection to LMStudio."""
    print("🧪 Testing LMStudio provider connection...")

    try:
        # Create provider using default configuration
        provider = create_provider("lmstudio")
        print(f"✅ Provider created: {provider.name}")
        print(f"   Default model: {provider.default_model}")
        print(f"   Tool support: {provider.supports_tools}")

        return provider
    except Exception as e:
        print(f"❌ Failed to create provider: {e}")
        return None


async def test_basic_chat(provider):
    """Test basic chat completion."""
    print("\n🗣️  Testing basic chat completion...")

    try:
        messages = [{"role": "user", "content": "Hello! Can you respond with a simple greeting?"}]

        print("   Sending message: 'Hello! Can you respond with a simple greeting?'")
        response = await provider.chat_complete(messages)

        print(f"✅ Response received:")
        print(f"   Content: {response.content}")
        print(f"   Finish reason: {response.finish_reason}")

        if response.usage:
            print(f"   Usage: {response.usage}")

        return True
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False


async def test_conversation(provider):
    """Test multi-turn conversation."""
    print("\n💬 Testing multi-turn conversation...")

    try:
        messages = [
            {"role": "user", "content": "What is 2 + 2?"},
        ]

        # First message
        print("   User: What is 2 + 2?")
        response1 = await provider.chat_complete(messages)
        print(f"   Assistant: {response1.content}")

        # Add response to conversation
        messages.append({"role": "assistant", "content": response1.content})
        messages.append({"role": "user", "content": "Now multiply that by 3"})

        # Second message
        print("   User: Now multiply that by 3")
        response2 = await provider.chat_complete(messages)
        print(f"   Assistant: {response2.content}")

        print("✅ Multi-turn conversation successful")
        return True
    except Exception as e:
        print(f"❌ Multi-turn conversation failed: {e}")
        return False


async def test_tool_calling_if_supported(provider):
    """Test tool calling if the provider supports it."""
    print("\n🔧 Testing tool calling...")

    if not provider.supports_tools:
        print("   ℹ️  Skipping tool test - provider doesn't support tools")
        return True

    try:
        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

        print("   Sending tool-enabled request...")
        response = await provider.chat_complete(messages, tools=tools)

        if response.tool_calls:
            print(f"✅ Tool call detected:")
            for tool_call in response.tool_calls:
                print(f"   - Function: {tool_call.name}")
                print(f"   - Arguments: {tool_call.arguments}")
        else:
            print("   ℹ️  No tool calls in response (model may not support function calling)")

        return True
    except Exception as e:
        print(f"❌ Tool calling test failed: {e}")
        return False


def print_configuration():
    """Print current configuration."""
    print("🔧 Current Configuration:")
    print(f"   LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'Not set')}")
    print(f"   LMSTUDIO_BASE_URL: {os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1 (default)')}")
    print(f"   LMSTUDIO_MODEL: {os.getenv('LMSTUDIO_MODEL', 'gpt-oss (default)')}")
    print(f"   LMSTUDIO_SUPPORTS_TOOLS: {os.getenv('LMSTUDIO_SUPPORTS_TOOLS', 'false (default)')}")


async def main():
    """Run all live tests."""
    print("🚀 LMStudio Provider Live Test")
    print("=" * 50)

    print_configuration()
    print()

    # Test 1: Connection
    provider = await test_lmstudio_connection()
    if not provider:
        print("\n❌ Cannot proceed without a working provider")
        return False

    # Test 2: Basic chat
    success = await test_basic_chat(provider)
    if not success:
        print("\n❌ Basic chat test failed")
        return False

    # Test 3: Multi-turn conversation
    success = await test_conversation(provider)
    if not success:
        print("\n❌ Conversation test failed")
        return False

    # Test 4: Tool calling (if supported)
    success = await test_tool_calling_if_supported(provider)
    if not success:
        print("\n❌ Tool calling test failed")
        return False

    print("\n" + "=" * 50)
    print("🎉 All tests passed! LMStudio provider is working correctly.")
    print("\nTo use the LMStudio provider in your code:")
    print("   provider = create_provider('lmstudio')")
    print("   response = await provider.chat_complete(messages)")

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)
