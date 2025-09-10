from __future__ import annotations
import copy
import json
from typing import Any

from lev.context_compressor import ContextCompressor
from lev.core.llm_provider import LlmProvider
from lev.prompts.judge import JUDGE_CRITIQUE_USER_PROMPT_TEMPLATE
from lev.scoring import Score, Scorer, ScoringContext

class LLMCritiqueScorer(Scorer):
    """LLM judge: did the assistant adequately answer the user?"""

    def __init__(
        self, llm_provider: LlmProvider, context_compressor: ContextCompressor, system_prompt: str | None = None
    ):
        self.llm_provider = llm_provider
        self.context_compressor = context_compressor
        self.system_prompt = system_prompt

    @property
    def display_name(self) -> str:
        return "LLM Critique Scorer"

    async def score(self, ctx: ScoringContext) -> Score:
        conversation = ctx.chat_history
        tool_calls = ctx.tool_calls

        user_messages = conversation.get_user_messages()
        if not user_messages:
            return Score(0.0, "No user query found")

        user_query = user_messages[0]
        distilled_conversation = conversation.render_trace()

        max_tokens = 10_000  # TODO: this should be known to the llm provider, but even if not, we can learn this from errors (or, just binary search)
        # Calculate remaining budget for tool calls after user query and conversation
        used_length = len(user_query.content) + len(distilled_conversation)
        remaining_budget = max_tokens - used_length - 200  # Reserve 200 chars for template overhead

        tool_calls_trace = self._serialize_tool_calls(tool_calls, max(0, remaining_budget))
        prompt = self._build_critic_prompt(user_query.content, distilled_conversation, tool_calls_trace)

        if len(prompt) > max_tokens:
            prompt = await self.context_compressor.compress_prompt(prompt)

        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            resp = await self.llm_provider.chat_complete(messages=messages)
            text = resp.content
            if not text:
                raise ValueError("Empty LLM result")

            result = json.loads(text)
            score_value = result.get("score", 0.0)
            justification = result.get("justification", "No justification provided")
            
            return Score(score_value, justification)
        except Exception as e:
            return Score(0.0, f"Evaluation failed: {e}")

    def _serialize_tool_calls(self, tool_calls: list[dict[str, Any]] | None, max_length: int) -> str:
        """
        Serialize tool calls to JSON, adapting to available budget.

        Args:
            tool_calls: List of tool call dictionaries
            max_length: Maximum character length allowed

        Returns:
            JSON string representation, potentially pruned to fit within max_length
        """
        if not tool_calls:
            return "None"

        # First attempt: serialize full tool calls
        full_json = json.dumps(tool_calls, indent=2)
        if len(full_json) <= max_length:
            return full_json

        # Second attempt: remove heavy response fields
        pruned_calls = []
        heavy_keys = {"response", "content", "tool_result", "result", "output", "return_value"}

        for call in tool_calls:
            pruned_call = copy.deepcopy(call)
            for key in heavy_keys:
                if key in pruned_call:
                    del pruned_call[key]
            pruned_calls.append(pruned_call)

        pruned_json = json.dumps(pruned_calls, indent=2)
        if len(pruned_json) <= max_length:
            return pruned_json

        # Third attempt: just function names and arguments
        function_calls = []
        for call in tool_calls:
            function_call = {
                "function": call.get("function"),
                "args": call.get("args"),
            }
            function_calls.append(function_call)

        function_calls_json = json.dumps(function_calls, indent=2)
        if len(function_calls_json) <= max_length:
            return function_calls_json

        # Final fallback: return summary
        return f"[Tool calls omitted: {len(tool_calls)} calls, {len(full_json)} chars over {max_length} limit]"

    def _build_critic_prompt(
        self, user_query: str, conversation_trace: str | None, tool_calls_trace: str | None = None
    ) -> str:
        return JUDGE_CRITIQUE_USER_PROMPT_TEMPLATE.format(
            user_query=user_query, conversation=conversation_trace, tool_calls_trace=tool_calls_trace
        )

def create_llm_critique_scorer(llm_provider: LlmProvider, context_compressor: ContextCompressor, system_prompt: str | None = None, **kwargs) -> LLMCritiqueScorer:
    """Factory function to create LLMCritiqueScorer."""
    return LLMCritiqueScorer(llm_provider, context_compressor, system_prompt)
