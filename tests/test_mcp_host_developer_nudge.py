import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lev.common.roles import MessageRole
from lev.core.agent import Agent
from lev.core.llm_provider import ModelResponse
from lev.host.mcp_host import McpHost, McpHostConfig
from lev.host.mcp_registry import McpClientRegistry


class MockAgent(Agent):
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self._is_initialized = True
        self.chat_history = MagicMock()
        self.chat_history.messages = []

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    async def message(self, message: str, tools=None, session=True, role=MessageRole.USER):
        response = (
            self.responses[self.call_count]
            if self.call_count < len(self.responses)
            else ModelResponse(content="default response")
        )
        self.call_count += 1
        return response

    async def reset(self):
        pass

    async def initialize(self):
        pass

    async def cleanup(self):
        pass


@pytest.mark.asyncio
async def test_developer_nudge_after_introspection():
    """Test that developer messages are injected when introspector says to continue."""

    # Mock responses
    agent_responses = [
        ModelResponse(content="I need more information"),  # First response
        ModelResponse(content="Final answer with nudge"),  # After developer nudge
    ]

    introspector_responses = [
        ModelResponse(
            content=json.dumps(
                {"continue": True, "reason": "Need more details", "next_prompt": "Please try a different approach"}
            )
        ),  # First introspection - continue with nudge
        ModelResponse(
            content=json.dumps({"continue": False, "reason": "Task complete"})
        ),  # Second introspection - stop
    ]

    # Setup mocks
    agent = MockAgent(agent_responses)
    introspector = MockAgent(introspector_responses)
    mcp_registry = MagicMock()
    mcp_registry.get_all_clients.return_value = []

    # Create host
    host = McpHost(agent=agent, mcp_registry=mcp_registry, introspector=introspector, config=McpHostConfig(max_steps=5))

    # Execute
    result = await host.prompt("Test question")

    # Verify
    assert result == "Final answer with nudge"
    assert agent.call_count == 2
    assert introspector.call_count == 2  # Called for both agent responses

    # Check journal for developer nudge
    journal_events = [entry["event"] for entry in host.journal]
    assert "introspect_prompt" in journal_events


@pytest.mark.asyncio
async def test_tool_error_introspection():
    """Test that tool errors are handled with introspection and developer nudges."""

    # Mock tool call response
    tool_response = ModelResponse(content="", tool_calls=[MagicMock(id="test_id", name="test_tool", arguments={})])

    agent_responses = [
        tool_response,  # Agent makes tool call
        ModelResponse(content="Final answer after error recovery"),  # After developer nudge
    ]

    introspector_responses = [
        ModelResponse(
            content=json.dumps(
                {
                    "continue": True,
                    "reason": "Tool failed, try alternative approach",
                    "next_prompt": "The tool failed. Please try a different method.",
                }
            )
        ),
        ModelResponse(content=json.dumps({"continue": False, "reason": "Task completed after recovery"})),
    ]

    # Setup mocks
    agent = MockAgent(agent_responses)
    agent.chat_history.add_assistant_tool_call_message = MagicMock()
    agent.chat_history.add_tool_response_message = MagicMock()
    agent.chat_history.messages = [{"role": "tool", "content": '{"success": false, "error": "Tool not found"}'}]

    introspector = MockAgent(introspector_responses)

    mcp_registry = MagicMock()
    mcp_registry.get_all_clients.return_value = []

    # Create host
    host = McpHost(agent=agent, mcp_registry=mcp_registry, introspector=introspector, config=McpHostConfig(max_steps=5))

    # Execute
    result = await host.prompt("Test question")

    # Verify
    assert result == "Final answer after error recovery"
    assert agent.call_count == 2
    assert introspector.call_count == 2  # Called after tool execution and after developer nudge

    # Check journal contains tool error handling
    journal_events = [entry["event"] for entry in host.journal]
    assert "execute_tools" in journal_events
    assert "introspect_prompt" in journal_events


@pytest.mark.asyncio
async def test_no_introspector_fallback():
    """Test that without an introspector, the host uses default behavior."""

    agent_responses = [
        ModelResponse(content="Simple answer"),
    ]

    agent = MockAgent(agent_responses)
    mcp_registry = MagicMock()
    mcp_registry.get_all_clients.return_value = []

    # Create host without introspector
    host = McpHost(
        agent=agent, mcp_registry=mcp_registry, introspector=None, config=McpHostConfig(max_steps=5)  # No introspector
    )

    # Execute
    result = await host.prompt("Test question")

    # Verify
    assert result == "Simple answer"
    assert agent.call_count == 1

    # Check journal shows no introspection attempts
    journal_events = [entry["event"] for entry in host.journal]
    assert "introspect_decision" in journal_events  # Default behavior still logs
    assert "introspect_prompt" not in journal_events  # But no developer nudges
