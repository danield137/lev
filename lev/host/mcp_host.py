import json
from typing import Any

from attr import dataclass

from lev.common.roles import MessageRole
from lev.core.agent import Agent
from lev.core.llm_provider import LlmProvider, ModelResponse, ToolCall
from lev.host.mcp_registry import McpClientRegistry


@dataclass(slots=True)
class Turn:
    content: str | None          # assistant text
    had_tools: bool              # whether tool_calls executed
    fatal_error: str | None = None

@dataclass(slots=True)
class McpHostConfig:
    max_steps: int = 8


class McpHost:
    agent: Agent
    mcp_registry: McpClientRegistry
    journal: list[dict[str, Any]]
    config: McpHostConfig
    introspector: Agent | None

    def __init__(
        self,
        agent: Agent,
        mcp_registry: McpClientRegistry,
        *,
        introspector: Agent | None = None,
        config: McpHostConfig | None = None,
    ):
        self.agent = agent
        self.mcp_registry = mcp_registry
        self.introspector = introspector
        self.config = config or McpHostConfig()
        self.journal: list[dict[str, Any]] = []  # audit trail

    async def reset(self) -> None:
        self.journal = []
        await self.agent.reset()
        await self.agent.initialize()

    async def warm_up(self) -> None:
        """
        Connect to all MCP clients and warm them up.
        """
        clients = self.mcp_registry.get_all_clients()
        for client in clients:
            await client.connect()

    async def step(self, prompt: str, *, role: MessageRole = MessageRole.USER) -> Turn:
        """
        Single round-trip:
        • send prompt/role to agent
        • execute any tool calls
        • return Turn(...)
        """
        try:
            if not self.agent.is_initialized:
                await self.agent.initialize()

            # Get agent response
            tools = await self._gather_tool_specs()
            model_resp = await self.agent.message(prompt, tools=tools, role=role)

            if not model_resp.tool_calls:
                # Plain answer, no tools
                self.agent.chat_history.add_assistant_message(model_resp.content or "")
                return Turn(content=model_resp.content, had_tools=False)

            # Execute tool calls
            await self._execute_tool_calls(model_resp)
            return Turn(content=model_resp.content, had_tools=True)

        except Exception as e:
            return Turn(content=None, had_tools=False, fatal_error=str(e))

    def history(self):
        """Simple accessor to chat history."""
        return self.agent.chat_history


    async def _gather_tool_specs(self) -> list[dict[str, Any]] | None:
        """
        Aggregate tool specifications from all connected MCP clients.
        """
        specs: list[dict[str, Any]] = []
        clients = self.mcp_registry.get_all_clients()

        for client in clients:
            try:
                if await client.is_connected():
                    client_specs = await client.get_tool_specs()
                    specs.extend(client_specs)
                else:
                    await client.connect()
                    client_specs = await client.get_tool_specs()
                    specs.extend(client_specs)
            except Exception as e:
                continue

        return specs if specs else None

    async def _execute_tool_calls(self, model_resp: ModelResponse) -> None:
        """
        Execute tool calls and add results to agent's chat history.
        """
        if not model_resp.tool_calls:
            return

        # Add assistant message with tool calls to history
        tool_calls_openai = self._to_openai_tool_calls(model_resp.tool_calls)
        self.agent.chat_history.add_assistant_tool_call_message(model_resp.content or "", tool_calls_openai)

        # Execute each tool call
        results = []
        for tool_call in model_resp.tool_calls:
            try:
                server_name = await self.mcp_registry.find_tool_server_name(tool_call.name)
                result = await self._execute_single_tool(tool_call)
                results.append({"tool": tool_call.name, "success": result.get("success", True)})
                self.agent.chat_history.add_tool_call(
                    server_name=server_name, tool_name=tool_call.name, arguments=tool_call.arguments, result=result
                )
                # Add tool response to chat history
                self.agent.chat_history.add_tool_response_message(tool_call.id, json.dumps(result))

            except Exception as e:
                error_result = {"success": False, "error": str(e)}
                results.append({"tool": tool_call.name, "success": False, "error": str(e)})

                # Add error response to chat history
                self.agent.chat_history.add_tool_response_message(tool_call.id, json.dumps(error_result))

    async def _execute_single_tool(self, tool_call: ToolCall) -> dict[str, Any]:
        """
        Execute a single tool call via the appropriate MCP client.
        """
        clients = self.mcp_registry.get_all_clients()

        for client in clients:
            try:
                available_tools = await client.list_tools()
                if tool_call.name in available_tools:
                    result = await client.call_tool(tool_call.name, tool_call.arguments)
                    return result
            except Exception as e:
                continue

        # Tool not found in any client
        error_msg = f"Tool {tool_call.name} not found in any connected MCP client"

        return {"success": False, "error": error_msg}

    def _to_openai_tool_calls(self, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """
        Convert ToolCall objects to OpenAI format for chat history.
        """
        return [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in tool_calls
        ]

    async def cleanup(self) -> None:
        """
        Clean up agent and MCP client connections.
        """
        await self.agent.cleanup()

        for client in self.mcp_registry.get_all_clients():
            try:
                await client.disconnect()
            except Exception:
                pass  # Ignore cleanup errors
