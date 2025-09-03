import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lev.agents.reasoning import ReasoningAgent
from lev.core.llm_provider import BaseLlmProvider, ModelResponse


class MockLlmProvider(BaseLlmProvider):
    def __init__(self, name: str = "mock", responses: list[ModelResponse] = None):
        super().__init__(name=name, supports_tools=True)
        self.responses = responses or []
        self.call_count = 0

    async def chat_complete(self, messages, tools=None):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        # Default response if no more responses
        return ModelResponse(content="Default response", tool_calls=None)


@pytest.mark.asyncio
async def test_introspection_answer_validation():
    """Test that answer validation works correctly."""

    # Mock main provider response
    main_response = ModelResponse(content="Brief answer", tool_calls=None)
    main_provider = MockLlmProvider("main", [main_response])

    # Mock inner provider validation responses
    validation_responses = [
        # First validation - invalid response
        ModelResponse(
            content='{"valid": false, "reason": "Insufficient detail", "followup_question": "Can you provide more details about X?"}',
            tool_calls=None,
        ),
        # Second validation - valid response
        ModelResponse(content='{"valid": true, "reason": "Response adequately answers the question"}', tool_calls=None),
    ]
    inner_provider = MockLlmProvider("inner", validation_responses)

    # Mock improved response after followup
    improved_response = ModelResponse(content="Detailed answer with more information", tool_calls=None)
    main_provider.responses.append(improved_response)

    agent = ReasoningAgent(llm_provider=main_provider, inner_provider=inner_provider, max_validation_attempts=2)

    # Mock the _get_tool_specs method
    agent._get_tool_specs = AsyncMock(return_value=[])

    result = await agent.message("Test question", track=False)

    # Should return the improved response after validation feedback
    assert result == "Detailed answer with more information"

    # Check that validation was called
    assert inner_provider.call_count == 2

    # Check that developer message was added to chat history
    developer_messages = [msg for msg in agent.chat_history.messages if msg.get("role") == "developer"]
    assert len(developer_messages) == 1
    assert developer_messages[0]["content"] == "Can you provide more details about X?"


@pytest.mark.asyncio
async def test_introspection_tool_failure_analysis():
    """Test that tool failure analysis works correctly."""

    # Mock provider that will suggest a tool call
    tool_call_mock = MagicMock()
    tool_call_mock.name = "test_tool"
    tool_call_mock.arguments = {"param": "value"}
    tool_call_mock.id = "call_123"

    main_response = ModelResponse(content="", tool_calls=[tool_call_mock])
    main_provider = MockLlmProvider("main", [main_response])

    # Mock inner provider analysis response - error is fixable
    analysis_response = ModelResponse(
        content='{"fixable": true, "suggestion": "Try different parameters", "retry_recommended": true}',
        tool_calls=None,
    )
    inner_provider = MockLlmProvider("inner", [analysis_response])

    agent = ReasoningAgent(llm_provider=main_provider, inner_provider=inner_provider, max_retries_per_call=1)

    # Mock tool specs and call_tool to fail first time
    agent._get_tool_specs = AsyncMock(return_value=[])

    call_count = 0

    async def mock_call_tool(name, args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call fails
            return {"success": False, "error": "Parameter error"}
        else:
            # Retry succeeds
            return {"success": True, "result": "Success"}

    agent.call_tool = AsyncMock(side_effect=mock_call_tool)

    # Mock _plan_retry to suggest a retry
    retry_tool_call = MagicMock()
    retry_tool_call.name = "test_tool"
    retry_tool_call.arguments = {"param": "fixed_value"}
    retry_tool_call.id = "call_124"

    retry_response = ModelResponse(content="", tool_calls=[retry_tool_call])
    agent._plan_retry = AsyncMock(return_value=retry_response)

    # Mock final response after tool execution
    final_response = ModelResponse(content="Task completed successfully", tool_calls=None)
    main_provider.responses.append(final_response)

    result = await agent.message("Execute tool", track=False)

    # Should complete successfully after retry
    assert result == "Task completed successfully"

    # Tool should have been called twice (original + retry)
    assert agent.call_tool.call_count == 2

    # Inner provider should have analyzed the failure
    assert inner_provider.call_count == 1


@pytest.mark.asyncio
async def test_direct_answer_without_tools():
    """Test that direct answers (no tools) still go through validation."""

    # Main provider gives direct answer
    main_response = ModelResponse(content="Direct answer", tool_calls=None)
    main_provider = MockLlmProvider("main", [main_response])

    # Inner provider validates it as good
    validation_response = ModelResponse(
        content='{"valid": true, "reason": "Response adequately answers the question"}', tool_calls=None
    )
    inner_provider = MockLlmProvider("inner", [validation_response])

    agent = ReasoningAgent(llm_provider=main_provider, inner_provider=inner_provider)

    agent._get_tool_specs = AsyncMock(return_value=[])

    result = await agent.message("Simple question", track=False)

    assert result == "Direct answer"
    assert inner_provider.call_count == 1  # Validation was called


if __name__ == "__main__":
    # Run a simple test
    asyncio.run(test_direct_answer_without_tools())
    print("Basic test passed!")
