import json
from typing import Any

from lev.core.agent import Agent
from lev.core.llm_provider import LlmProvider, ModelResponse, ToolCall
from lev.host.mcp_registry import McpClientRegistry


class McpHost:
    """
    MCP Host that orchestrates agent proposals and tool execution.
    
    Stages:
    1. Propose - agent suggests response or tool calls
    2. Execute - host executes tools via MCP clients  
    3. Introspect - optional reasoning about whether to continue
    """

    def __init__(
        self,
        agent: Agent,
        mcp_registry: McpClientRegistry,
        *,
        introspector: Agent | None = None,
        max_steps: int = 8,
    ):
        self.agent = agent
        self.mcp_registry = mcp_registry
        self.introspector = introspector
        self.max_steps = max_steps
        self.journal: list[dict[str, Any]] = []  # audit trail

    async def prompt(self, question: str) -> str:
        """
        Main entry point. Orchestrates the full agent + tool loop.
        """
        depth = 0
        await self.agent.reset()
        await self.agent.initialize()
        
        # Clear journal for new conversation
        self.journal = []
        self._log_journal("start", {"question": question, "max_steps": self.max_steps})

        while depth < self.max_steps:
            depth += 1
            self._log_journal("step", {"depth": depth})

            # ---------- 1. Propose stage ----------
            tools = await self._gather_tool_specs()
            model_resp = await self.agent.message(question, tools=tools)
            
            if not model_resp.tool_calls:
                # Plain answer - we're done
                answer = model_resp.content or "No response generated."
                self._log_journal("propose", {"type": "answer", "content_length": len(answer)})
                
                # Optional introspection gate
                if await self._introspect("answer"):
                    self._log_journal("introspect", {"action": "continue"})
                    continue
                    
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
            if await self._introspect("tool_execution"):
                self._log_journal("introspect", {"action": "continue"})
                continue

            # If introspector says we're done, get final answer
            break

        # Max steps reached or introspector decided to stop
        final_msg = self.agent.chat_history.messages[-1] if self.agent.chat_history.messages else None
        final_answer = final_msg.get("content", "step budget exceeded") if final_msg else "step budget exceeded"
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
                result = await self._execute_single_tool(tool_call)
                results.append({"tool": tool_call.name, "success": result.get("success", True)})
                
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

    async def _introspect(self, stage: str) -> bool:
        """
        Optional introspection step using introspector agent.
        Returns True if the host should continue the loop.
        """
        if not self.introspector:
            # No introspector - default behavior based on stage
            if stage == "answer":
                return False  # Stop on plain answer
            return False  # Stop after tool execution by default

        try:
            # Get conversation history for introspection
            history_summary = self._summarize_history()
            
            prompt = f"""
            Analyze this conversation and decide the next action.
            
            Stage: {stage}
            
            Conversation summary:
            {history_summary}
            
            Should the host continue the conversation loop?
            Respond with JSON: {{"continue": true/false, "reason": "explanation"}}
            """
            
            introspect_resp = await self.introspector.message(prompt, track=False)
            
            try:
                decision = json.loads(introspect_resp.content or "{}")
                should_continue = decision.get("continue", False)
                reason = decision.get("reason", "no reason provided")
                
                self._log_journal("introspect_decision", {
                    "stage": stage,
                    "continue": should_continue,
                    "reason": reason
                })
                
                return should_continue
                
            except json.JSONDecodeError:
                # Fallback if introspector doesn't return valid JSON
                self._log_journal("introspect_error", {"stage": stage, "error": "invalid_json"})
                return False
                
        except Exception as e:
            self._log_journal("introspect_error", {"stage": stage, "error": str(e)})
            return False

    def _summarize_history(self) -> str:
        """
        Create a summary of the conversation history for introspection.
        """
        messages = self.agent.chat_history.messages
        if not messages:
            return "No conversation history"
        
        summary_parts = []
        for msg in messages[-5:]:  # Last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)

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
