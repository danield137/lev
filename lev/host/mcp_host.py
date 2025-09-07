import json
from typing import Any

from attr import dataclass

from lev.common.roles import MessageRole
from lev.core.agent import Agent
from lev.core.llm_provider import LlmProvider, ModelResponse, ToolCall
from lev.host.mcp_registry import McpClientRegistry
from lev.prompts.reasoning import REASONING_AGENT_INTROSPECTIVE_TEMPLATE, REASONING_AGENT_ANSWER_VALIDATION_PROMPT


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

    async def prompt(self, question: str) -> str:
        if not self.agent.is_initialized:
            await self.agent.initialize()

        depth = 0

        # Send initial question
        prompt_text = question
        role = MessageRole.USER
        model_resp = ModelResponse.empty()
        while depth < self.config.max_steps:
            depth += 1
            # ---------- 1. Propose stage ----------
            tools = await self._gather_tool_specs()
            model_resp = await self.agent.message(prompt_text, tools=tools, role=role)

            if not model_resp.tool_calls:
                # Plain answer - check if we should continue
                answer = model_resp.content or "No response generated."

                # Make sure the answer is complete
                decision = await self._introspect(maybe_done=True)
                if decision["continue"]:
                    role = MessageRole.DEVELOPER
                    # Inject developer message if provided
                    prompt_text = decision.get("next_prompt") or decision.get("reason")
                    continue  # Continue the loop with developer message
                return answer

            # ---------- 2. Execute stage ----------
            await self._execute_tool_calls(model_resp)

            # ---------- 3. Introspect stage ----------
            decision = await self._introspect()
            prompt_text = "Proceed to provide the final answer."
            role = MessageRole.DEVELOPER
            if decision["continue"]:
                # Inject developer message if provided, else request final answer
                next_prompt = decision.get("next_prompt")

                if next_prompt:
                    prompt_text = next_prompt
                    
                continue
            else:
                continue

        if model_resp.content is None:
            print(self.agent.chat_history.render_trace())
        # Max steps reached or introspector decided to stop
        final_answer = model_resp.content or "No final answer generated."

        return final_answer

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


    async def _introspect(self, maybe_done: bool = False) -> dict[str, Any]:
        """
        Optional introspection step using introspector agent.
        If maybe_done=True, uses answer validation prompt. Otherwise uses default introspection.
        Returns decision dict with continue, reason, and optional next_prompt.
        """
        if not self.introspector:
            # No introspector - default behavior
            result = {"continue": False, "reason": "no introspector, stopping"}
            return result

        try:
            # Get conversation history for introspection
            history_summary = self.agent.chat_history.render_trace()
            
            if maybe_done:
                # Use answer validation prompt
                answer = self.agent.chat_history.messages[-1].get("content", "") if self.agent.chat_history.messages else ""
                prompt = REASONING_AGENT_ANSWER_VALIDATION_PROMPT.format(
                    conversation_history=history_summary,
                    response_to_validate=answer,
                )
            else:
                # Use default introspection
                prompt = history_summary
                
            await self.introspector.reset()  # kind of a hack. we need the system prompt, but not the conversation at this point
            introspect_resp = await self.introspector.message(prompt)  # run session-less, so we don't mistakenly include other messages

            try:
                decision = json.loads(introspect_resp.content or "{}")
                
                if maybe_done:
                    # For answer validation, check if valid
                    if not decision.get("valid", True):
                        followup = decision.get("followup_question", "Please provide more details.")
                        return {"continue": True, "reason": "answer validation failed", "next_prompt": followup}
                    else:
                        return {"continue": False, "reason": "answer validation passed"}
                else:
                    # Default introspection behavior
                    should_continue = decision.get("continue", False)
                    reason = decision.get("reason", "no reason provided")
                    next_prompt = decision.get("next_prompt")
                    return {"continue": should_continue, "reason": reason, "next_prompt": next_prompt}

            except json.JSONDecodeError:
                # Fallback if introspector doesn't return valid JSON
                return {"continue": False, "reason": "introspector returned invalid JSON"}

        except Exception as e:
            return {"continue": False, "reason": f"introspection failed: {str(e)}"}

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
