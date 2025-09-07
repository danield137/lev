import json
from typing import Any

from attr import dataclass

from lev.common.roles import MessageRole
from lev.core.agent import Agent
from lev.core.llm_provider import LlmProvider, ModelResponse, ToolCall
from lev.host.mcp_registry import McpClientRegistry
from lev.prompts.reasoning import REASONING_AGENT_INTROSPECTIVE_TEMPLATE


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
        # Clear journal for new conversation
        self._log_journal("question", {"question": question})
        
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
                self._log_journal("propose", {"type": "answer", "content_length": len(answer)})
                
                # Optional introspection gate
                decision = await self._introspect("answer")
                if decision["continue"]:
                    self._log_journal("introspect", {"action": "continue"})
                    # Inject developer message if provided
                    dev_msg = decision.get("next_prompt") or decision.get("reason")
                    if dev_msg:
                        prompt_text = dev_msg
                        role = MessageRole.DEVELOPER
                        self._log_journal("introspect_prompt", {"message": dev_msg})
                        continue  # Continue the loop with developer message
                else:
                    self._log_journal("introspect", {"action": "stop", "reason": decision.get("reason")})
                    
                # Introspector says stop or no next_prompt - return answer
                self._log_journal("complete", {"final_answer": True})
                return answer

            # Agent proposed tool calls
            self._log_journal("propose", {
                "type": "tool_calls", 
                "count": len(model_resp.tool_calls),
                "tools": [tc.name for tc in model_resp.tool_calls]
            })

            # ---------- 2. Execute stage ----------
            await self._execute_tool_calls(model_resp)

            # ---------- 3. Introspect stage ---------- 
            decision = await self._introspect("tool_execution")
            if decision["continue"] and decision.get("next_prompt"):
                # Inject developer message if provided
                next_prompt = decision.get("next_prompt")
                if next_prompt:
                    prompt_text = next_prompt
                    role = MessageRole.DEVELOPER
                    self._log_journal("introspect_prompt", {"message": next_prompt})
                    continue
            else:
                self._log_journal("introspect", {"action": "stop", "reason": decision.get("reason")})
                break

        # Max steps reached or introspector decided to stop
        final_answer = model_resp.content or "No final answer generated."
        self._log_journal("complete", {"reason": "max_steps_or_introspector", "final_answer": len(final_answer)})
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
                self._log_journal("tool_spec_error", {
                    "client": client.name, 
                    "error": str(e)
                })
                continue

        self._log_journal("gather_tools", {"total_specs": len(specs)})
        return specs if specs else None

    async def _execute_tool_calls(self, model_resp: ModelResponse) -> None:
        """
        Execute tool calls and add results to agent's chat history.
        """
        if not model_resp.tool_calls:
            return

        # Add assistant message with tool calls to history
        tool_calls_openai = self._to_openai_tool_calls(model_resp.tool_calls)
        self.agent.chat_history.add_assistant_tool_call_message(
            model_resp.content or "", tool_calls_openai
        )

        # Execute each tool call
        results = []
        for tool_call in model_resp.tool_calls:
            try:
                server_name = await self.mcp_registry.find_tool_server_name(tool_call.name)
                result = await self._execute_single_tool(tool_call)
                results.append({"tool": tool_call.name, "success": result.get("success", True)})
                self.agent.chat_history.add_tool_call(server_name=server_name, tool_name=tool_call.name, arguments=tool_call.arguments, result=result)
                # Add tool response to chat history
                self.agent.chat_history.add_tool_response_message(
                    tool_call.id, json.dumps(result)
                )
                
            except Exception as e:
                error_result = {"success": False, "error": str(e)}
                results.append({"tool": tool_call.name, "success": False, "error": str(e)})
                
                # Add error response to chat history  
                self.agent.chat_history.add_tool_response_message(
                    tool_call.id, json.dumps(error_result)
                )

        self._log_journal("execute_tools", {"results": results})

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
                    self._log_journal("tool_call", {
                        "client": client.name,
                        "tool": tool_call.name,
                        "success": result.get("success", True)
                    })
                    return result
            except Exception as e:
                self._log_journal("tool_error", {
                    "client": client.name,
                    "tool": tool_call.name, 
                    "error": str(e)
                })
                continue

        # Tool not found in any client
        error_msg = f"Tool {tool_call.name} not found in any connected MCP client"
        self._log_journal("tool_not_found", {"tool": tool_call.name})
        return {"success": False, "error": error_msg}

    async def _introspect(self, stage: str) -> dict[str, Any]:
        """
        Optional introspection step using introspector agent.
        Returns decision dict with continue, reason, and optional next_prompt.
        """
        if not self.introspector:
            # No introspector - default behavior based on stage
            result = {"continue": False, "reason": "no introspector, stopping on answer"}
            self._log_journal("introspect_decision", {
                "stage": stage,
                "continue": False,
                "reason": result["reason"],
                "next_prompt": None
            })
            return result

        try:
            # Get conversation history for introspection
            history_summary = self.agent.chat_history.render_trace()
            await self.introspector.reset() # kind of a hack. we need the system prompt, but not the conversation at this point
            introspect_resp = await self.introspector.message(history_summary) # run session-less, so we don't mistakenly include other messages

            try:
                decision = json.loads(introspect_resp.content or "{}")
                should_continue = decision.get("continue", False)
                reason = decision.get("reason", "no reason provided")
                next_prompt = decision.get("next_prompt")

                result = {
                    "continue": should_continue,
                    "reason": reason,
                    "next_prompt": next_prompt
                }
                
                self._log_journal("introspect_decision", {
                    "stage": stage,
                    "continue": should_continue,
                    "reason": reason,
                    "next_prompt": next_prompt
                })
                
                return result
                
            except json.JSONDecodeError:
                # Fallback if introspector doesn't return valid JSON
                self._log_journal("introspect_error", {"stage": stage, "error": "invalid_json"})
                return {"continue": False, "reason": "introspector returned invalid JSON"}
                
        except Exception as e:
            self._log_journal("introspect_error", {"stage": stage, "error": str(e)})
            return {"continue": False, "reason": f"introspection failed: {str(e)}"}

    def _to_openai_tool_calls(self, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """
        Convert ToolCall objects to OpenAI format for chat history.
        """
        return [
            {
                "id": tc.id,
                "type": "function", 
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments)
                }
            }
            for tc in tool_calls
        ]

    def _log_journal(self, event: str, data: dict[str, Any]) -> None:
        """
        Log events to the audit journal.
        """
        self.journal.append({
            "event": event,
            "data": data
        })

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
