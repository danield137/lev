import json
from typing import Any

from lev.common.roles import MessageRole
from lev.core.agent import Agent
from lev.core.llm_provider import LlmProvider, ModelResponse
from lev.host.mcp_client import McpClient
from lev.llm_providers.provider_factory import create_tool_enabled_provider
from lev.prompts.tool_agent import TOOL_AGENT_DEFAULT_SYSTEM_PROMPT


class ToolsAgent(Agent):
    mcp_clients: list[McpClient]
    find_errors_in_content: bool = True
    _is_initialized: bool = False

    def __init__(
        self,
        llm_provider: LlmProvider,
        system_prompt: str,
        mcp_clients: list[McpClient] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        llm_provider = llm_provider or create_tool_enabled_provider()
        self.mcp_clients = mcp_clients or []
        if not llm_provider.supports_tools:
            raise ValueError(f"Provider {llm_provider.name} does not support tool calling")
        system_prompt = system_prompt or TOOL_AGENT_DEFAULT_SYSTEM_PROMPT
        super().__init__(llm_provider, system_prompt, temperature=temperature, max_tokens=max_tokens)

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    async def initialize(self) -> None:
        for c in self.mcp_clients:
            await c.connect()
        self._is_initialized = True

    async def cleanup(self) -> None:
        for c in self.mcp_clients:
            await c.disconnect()
        self._is_initialized = False

    async def message(
        self,
        message: str,
        tools: list[dict[str, Any]] | None = None,
        session: bool = True,
        role: MessageRole = MessageRole.USER,
    ) -> ModelResponse:
        self.chat_history.add_message(message, role=role)

        # Use provided tools or get tool specs from connected clients
        if tools is None:
            tools = await self._get_tool_specs()

        # Get messages for the LLM
        messages = self.chat_history.to_role_content_messages(with_system=True, with_tools=True)
        if not session:
            messages = [{"role": role.value, "content": message}]
        # Call the LLM with tools - return raw ModelResponse
        return await self.llm_provider.chat_complete(messages, tools=tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        for c in self.mcp_clients:
            try:
                if tool_name in await c.list_tools():
                    result = await c.call_tool(tool_name, arguments)
                    # FIXME: temporary workaround for tool response errors
                    if (
                        self.find_errors_in_content
                        and "content" in result
                        and result["content"][:10].lower().startswith("error")
                    ):
                        result = {"success": False, "error": result["content"]}
                    self.chat_history.add_tool_call(c.server_name, tool_name, arguments, result)
                    return result
            except Exception as e:
                err = {"success": False, "error": str(e)}
                self.chat_history.add_tool_call(c.server_name, tool_name, arguments, err)
                return err
        return {"success": False, "error": f"Tool {tool_name} not found in any connected MCP client"}

    async def _answer_with_tools(self) -> str:
        try:
            tools = await self._get_tool_specs()
            first = await self.llm_provider.chat_complete(
                messages=self.chat_history.to_role_content_messages(with_system=True), tools=tools
            )
            if not first.tool_calls:
                return first.content or "No response generated."

            self.chat_history.add_assistant_tool_call_message(
                first.content or "", self._to_openai_tool_calls(first.tool_calls)
            )

            await self._exec_tool_calls(first.tool_calls)
            return await self._finalize(tools)
        except Exception as e:
            return f"Error: {e}"

    async def _exec_tool_calls(self, tool_calls) -> None:
        for tc in tool_calls:
            try:
                res = await self.call_tool(tc.name, tc.arguments)
            except Exception as e:
                res = {"success": False, "error": str(e)}
            self.chat_history.add_tool_response_message(tc.id, json.dumps(res))

    async def _finalize(self, tools) -> str:
        final = await self.llm_provider.chat_complete(messages=self._build_messages(), tools=tools)
        return final.content or "Processed with tools."

    def _build_messages(self) -> list[dict[str, Any]]:
        system = self.system_prompt
        msgs = [{"role": "system", "content": system}]
        for m in self.chat_history:
            entry = {"role": m["role"]}
            if "content" in m:
                entry["content"] = m["content"]
            if "tool_calls" in m:
                entry["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                entry["tool_call_id"] = m["tool_call_id"]
            msgs.append(entry)
        return msgs

    def _to_openai_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        return [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in tool_calls
        ]

    async def _get_tool_specs(self) -> list[dict[str, Any]] | None:
        specs: list[dict[str, Any]] = []
        for c in self.mcp_clients:
            try:
                if await c.is_connected():
                    specs.extend(await c.get_tool_specs())
            except Exception:
                pass
        return specs or None
