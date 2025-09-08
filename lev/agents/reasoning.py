import json
from datetime import datetime
from typing import Any, Optional

from lev.agents.tool import ToolsAgent
from lev.common.roles import MessageRole
from lev.core.chat_history import ChatHistory
from lev.core.llm_provider import LlmProvider, ModelResponse
from lev.prompts.reasoning import (
    REASONING_AGENT_ANSWER_VALIDATION_PROMPT,
    REASONING_AGENT_DEFAULT_SYSTEM_PROMPT,
    REASONING_AGENT_INTROSPECTIVE_TEMPLATE,
    REASONING_AGENT_RETRY_PROMPT,
    REASONING_AGENT_TOOL_FAILURE_ANALYSIS_PROMPT,
)


class ReasoningAgent(ToolsAgent):
    """
    A reasoning agent is a tool agent that also has an internal monologue, and can reason about next steps.

    Behavior:
    - Uses introspection as a gate before returning results
    - Validates assistant replies to ensure they answer the question
    - Analyzes failed tool calls to determine if they can be fixed
    - Uses separate inner_provider for introspection if provided
    """

    max_steps: int
    max_retries_per_call: int
    max_validation_attempts: int
    inner_provider: LlmProvider

    def __init__(
        self,
        llm_provider: LlmProvider,
        inner_provider: Optional[LlmProvider] = None,
        system_prompt: Optional[str] = None,
        mcp_clients: Optional[list] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_steps: int = 8,
        max_retries_per_call: int = 2,
        max_validation_attempts: int = 3,
    ):
        system_prompt = system_prompt or REASONING_AGENT_DEFAULT_SYSTEM_PROMPT
        super().__init__(
            llm_provider,
            system_prompt,
            mcp_clients,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_validation_attempts = max_validation_attempts
        self.inner_provider = inner_provider or llm_provider

    async def message(
        self,
        message: str,
        tools: list[dict[str, Any]] | None = None,
        session: bool = True,
        role: MessageRole = MessageRole.USER,
    ) -> ModelResponse:
        """
        Orchestrates reasoning with introspection gates:
          1) Get initial response (with or without tools)
          2) Execute tools if needed with failure analysis
          3) Validate final answer before returning
          4) Ask developer followups if answer is insufficient
        """
        self.chat_history.add_user_message(message)

        try:
            # Use provided tools or get tool specs from connected clients
            if tools is None:
                tools = await self._get_tool_specs()

            step_count = 0

            # Get initial response
            response = await self.llm_provider.chat_complete(
                messages=self.chat_history.to_role_content_messages(with_system=True),
                tools=tools,
            )

            # Execute tools with step counting to prevent infinite loops
            while response.tool_calls and step_count < self.max_steps:
                step_count += 1
                response = await self._execute_tools_with_introspection(response, tools)

            # For now, return the response directly - introspection will be handled by McpHost
            self.chat_history.add_assistant_message(response.content)  # type: ignore

            return response

        except Exception as e:
            error_response = ModelResponse(content=f"Error: {e}")
            self.chat_history.add_assistant_message(error_response.content or "Unknown error occurred")
            return error_response

    # ============= Planning helpers (explicit names) =============

    async def _plan_decompose(self, tools: Optional[list[dict[str, Any]]]) -> ModelResponse:
        """
        Optional helper if a separate decomposition turn is needed.
        Not used directly in the main loop because we start with the first tools-enabled call.
        """
        return await self.llm_provider.chat_complete(
            messages=self.chat_history.to_role_content_messages(with_system=True),
            tools=tools,
        )

    async def _plan_introspect(self, tools: Optional[list[dict[str, Any]]], history: ChatHistory) -> ModelResponse:
        """
        After tool execution, decide: finish with an answer (no tool_calls) or propose next tool call(s).
        Uses REASONING_AGENT_INTROSPECTIVE_TEMPLATE and allows tools in this call.
        """
        conv = history.render_trace()
        sys = REASONING_AGENT_INTROSPECTIVE_TEMPLATE.format(conversation_history=conv)
        return await self.llm_provider.chat_complete(
            messages=[{"role": "system", "content": sys}],
            tools=tools,
        )

    async def _plan_retry(self, tools: Optional[list[dict[str, Any]]], history: ChatHistory) -> ModelResponse:
        """
        When a tool call fails, ask the model to propose a corrected tool invocation or explain the error.
        Uses REASONING_AGENT_RETRY_PROMPT and allows tools in this call.
        """
        conv = history.render_trace()
        sys = REASONING_AGENT_RETRY_PROMPT.format(conversation_history=conv)
        return await self.llm_provider.chat_complete(
            messages=[{"role": "system", "content": sys}],
            tools=tools,
        )

    # ============= Introspection gates =============

    async def _introspect_answer(self, response: str) -> str:
        """
        Gate: Validates if the response adequately answers the user's question.
        If not, asks developer followup questions until satisfied or max attempts reached.
        """
        validation_attempts = 0
        current_response = response

        while validation_attempts < self.max_validation_attempts:
            try:
                conv = self.chat_history.render_trace()
                prompt = REASONING_AGENT_ANSWER_VALIDATION_PROMPT.format(
                    conversation_history=conv, response_to_validate=current_response
                )

                validation = await self.inner_provider.chat_complete(messages=[{"role": "system", "content": prompt}])

                # Parse validation response
                validation_result = self._parse_json_response(validation.content)

                if validation_result.get("valid", True):
                    # Response is satisfactory
                    return current_response

                # Response is insufficient, ask followup question with developer role
                followup_question = validation_result.get("followup_question", "Please provide more details.")

                # Add developer followup to chat history
                self.chat_history.messages.append(
                    {"role": "developer", "content": followup_question, "timestamp": datetime.now().isoformat()}
                )

                # Get improved response from main provider
                tools = await self._get_tool_specs()
                improved = await self.llm_provider.chat_complete(
                    messages=self.chat_history.to_role_content_messages(with_system=True, with_tools=True),
                    tools=tools,
                )

                if improved.tool_calls:
                    # Execute tools if needed
                    improved = await self._execute_tools_with_introspection(improved, tools)

                current_response = improved.content or current_response
                validation_attempts += 1

            except Exception as e:
                # If introspection fails, return original response
                return response

        # Max validation attempts reached
        return current_response

    async def _introspect_tool_failure(
        self, tool_name: str, tool_arguments: dict[str, Any], error_message: str
    ) -> dict[str, Any]:
        """
        Gate: Analyzes tool call failures to determine if they can be fixed.
        Returns analysis with fixable status and suggestions.
        """
        try:
            conv = self.chat_history.render_trace()
            prompt = REASONING_AGENT_TOOL_FAILURE_ANALYSIS_PROMPT.format(
                conversation_history=conv,
                tool_name=tool_name,
                tool_arguments=json.dumps(tool_arguments),
                error_message=error_message,
            )

            analysis = await self.inner_provider.chat_complete(messages=[{"role": "system", "content": prompt}])

            return self._parse_json_response(analysis.content or "{}")

        except Exception:
            # If analysis fails, assume not fixable
            return {"fixable": False, "reason": "Analysis failed", "retry_recommended": False}

    async def _execute_tools_with_introspection(
        self, response: ModelResponse, tools: Optional[list[dict[str, Any]]]
    ) -> ModelResponse:
        """
        Execute tool calls with introspection on failures.
        Uses the introspection gate to analyze failures and decide on retries.
        """
        if not response.tool_calls:
            return response

        # Record assistant tool calls
        self.chat_history.add_assistant_tool_call_message(
            response.content or "",
            self._to_openai_tool_calls(response.tool_calls),
        )

        # Execute each tool call with introspection
        for tc in response.tool_calls:
            success = await self._execute_single_tool_with_introspection(tc, tools)
            if not success:
                # Tool failed and couldn't be recovered, continue with next tool or finish
                pass

        # Get final response after tool execution (include tool responses in conversation)
        final_response = await self.llm_provider.chat_complete(
            messages=self.chat_history.to_role_content_messages(with_system=True, with_tools=True),
            tools=tools,
        )

        return final_response

    async def _execute_single_tool_with_introspection(
        self, tool_call: Any, tools: Optional[list[dict[str, Any]]]
    ) -> bool:
        """
        Execute a single tool call with introspection on failure.
        Returns True if successful, False if failed after retries.
        """
        # First attempt
        try:
            res = await self.call_tool(tool_call.name, tool_call.arguments)
        except Exception as e:
            res = {"success": False, "error": str(e)}

        self.chat_history.add_tool_response_message(tool_call.id, json.dumps(res))

        if res.get("success", False):
            return True

        # Tool failed, use introspection to analyze
        error_message = res.get("error", "Unknown error")
        analysis = await self._introspect_tool_failure(tool_call.name, tool_call.arguments, error_message)

        if not analysis.get("fixable", False) or not analysis.get("retry_recommended", False):
            # Error is not fixable, give up
            return False

        # Try to retry with fixes
        retries = 0
        while retries < self.max_retries_per_call:
            retries += 1

            # Ask for corrected tool call
            retry_resp = await self._plan_retry(tools, self.chat_history)

            if retry_resp.tool_calls:
                # Execute the corrected tool call
                self.chat_history.add_assistant_tool_call_message(
                    retry_resp.content or "",
                    self._to_openai_tool_calls(retry_resp.tool_calls),
                )

                r_tc = retry_resp.tool_calls[0]
                try:
                    r_res = await self.call_tool(r_tc.name, r_tc.arguments)
                except Exception as e:
                    r_res = {"success": False, "error": str(e)}

                self.chat_history.add_tool_response_message(r_tc.id, json.dumps(r_res))

                if r_res.get("success", False):
                    return True

                # Still failed, analyze again
                r_error = r_res.get("error", "Unknown error")
                r_analysis = await self._introspect_tool_failure(r_tc.name, r_tc.arguments, r_error)

                if not r_analysis.get("retry_recommended", False):
                    # Stop retrying
                    break
            else:
                # No retry suggested
                break

        return False

    def _parse_json_response(self, content: str | None) -> dict[str, Any]:
        """Helper to parse JSON responses from introspection calls."""
        if not content:
            return {}
        try:
            return json.loads(content.strip())
        except (json.JSONDecodeError, AttributeError):
            return {}

    # ============= Tool execution with retry (legacy) =============

    async def _exec_with_retries(
        self, tool_calls: list[Any], tools: Optional[list[dict[str, Any]]]
    ) -> tuple[bool, str]:
        """
        Legacy method - kept for compatibility but no longer used in main flow.
        Execute tool calls in sequence. On failure, try to re-plan a corrected call up to max_retries_per_call.
        Returns (finished, final_message). If finished is True, the agent is done and final_message is the answer.
        """
        for tc in tool_calls:
            # First attempt
            try:
                res = await self.call_tool(tc.name, tc.arguments)
            except Exception as e:
                res = {"success": False, "error": str(e)}
            self.chat_history.add_tool_response_message(tc.id, json.dumps(res))

            if res.get("success", False):
                # Successful tool call; continue to (potentially) more calls or next introspection
                continue

            # Retry path
            retries = 0
            while retries < self.max_retries_per_call:
                retries += 1
                retry_resp = await self._plan_retry(tools, self.chat_history)

                if retry_resp.tool_calls:
                    # Execute the new proposed tool call(s)
                    self.chat_history.add_assistant_tool_call_message(
                        retry_resp.content or "",
                        self._to_openai_tool_calls(retry_resp.tool_calls),
                    )
                    # Execute only the first proposed call at a time to maintain control
                    r_tc = retry_resp.tool_calls[0]
                    try:
                        r_res = await self.call_tool(r_tc.name, r_tc.arguments)
                    except Exception as e:
                        r_res = {"success": False, "error": str(e)}
                    self.chat_history.add_tool_response_message(r_tc.id, json.dumps(r_res))

                    if r_res.get("success", False):
                        # Retry succeeded; proceed to next tool or introspection
                        break
                    # else loop for another retry attempt
                else:
                    # No tool call proposed; the model provided an explanation or answer.
                    content = (retry_resp.content or "").strip()
                    if content:
                        return True, content
                    # If no content and no tool_calls, abort retry loop.
                    break

            # Exhausted retries without success – continue to introspection to decide next step.
        return False, ""
