import copy
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union

import numpy as np

from lev.context_compressor import ContextCompressor
from lev.core.chat_history import ChatHistory
from lev.core.llm_provider import LlmProvider
from lev.llm_providers import create_provider
from lev.prompts.judge import JUDGE_CRITIQUE_USER_PROMPT_TEMPLATE, JUDGE_EXTRACT_USER_PROMPT_TEMPLATE


# === Typed inputs ===
@dataclass
class MatchInput:
    raw_input: str
    expected: dict[str, Any]


@dataclass
class CritiqueInput:
    conversation: ChatHistory
    tool_calls: Optional[List[dict[str, Any]]] = None


@dataclass
class ExtractInput:
    conversation: ChatHistory
    expected: Any


InputType = TypeVar("InputType")


# === Base interface ===
class Scorer(ABC, Generic[InputType]):
    @abstractmethod
    async def score(self, inputs: InputType) -> dict[str, Any]:
        pass


# === Scorers ===
class MatchScorer(Scorer[MatchInput]):
    """Heuristic scoring by relevance, completeness, accuracy."""

    async def score(self, inputs: MatchInput) -> dict[str, Any]:
        rel = self._score_relevance(inputs.raw_input, inputs.expected)
        comp = self._score_completeness(inputs.raw_input, inputs.expected)
        acc = self._score_accuracy(inputs.raw_input, inputs.expected)
        overall = np.mean([rel, comp, acc])

        return {
            "relevance": rel,
            "completeness": comp,
            "accuracy": acc,
            "overall": overall,
            "mode": EvaluationMode.MATCH.value,
        }

    def _score_relevance(self, message: str, expected: dict[str, Any]) -> float:
        topic: str = expected.get("topic", "")
        if topic and topic.lower() in message.lower():
            return min(1.0, 0.8 + len(message) / 5000)
        return 0.5

    def _score_completeness(self, message: str, expected: dict[str, Any]) -> float:
        points: list[str] = expected.get("key_points", [])
        if not points:
            return 0.7
        found = sum(p.lower() in message.lower() for p in points)
        return min(1.0, found / len(points))

    def _score_accuracy(self, message: str, expected: dict[str, Any]) -> float:
        indicators: list[str] = expected.get("accuracy_indicators", [])
        if not indicators:
            return 0.7
        found = sum(i.lower() in message.lower() for i in indicators)
        return min(1.0, found / len(indicators))


class CritiqueScorer(Scorer[CritiqueInput]):
    """LLM judge: did the assistant adequately answer the user?"""

    llm_provider: LlmProvider
    context_compressor: ContextCompressor
    system_prompt: Optional[str]

    def __init__(
        self, llm_provider: LlmProvider, context_compressor: ContextCompressor, system_prompt: Optional[str] = None
    ):
        self.llm_provider = llm_provider
        self.context_compressor = context_compressor
        self.system_prompt = system_prompt

    async def score(self, inputs: CritiqueInput) -> dict[str, Any]:
        user_query = inputs.conversation.get_user_messages()[0]
        distilled_conversation = inputs.conversation.render_trace()
        tool_calls = inputs.tool_calls

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
            result["mode"] = EvaluationMode.CRITIQUE.value
            return result
        except Exception as e:
            return {
                "answered": False,
                "score": 0.0,
                "justification": f"Evaluation failed: {e}",
                "mode": EvaluationMode.CRITIQUE.value,
            }

    def _serialize_tool_calls(self, tool_calls: Optional[List[dict[str, Any]]], max_length: int) -> str:
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
        self, user_query: str, conversation_trace: Optional[str], tool_calls_trace: Optional[str] = None
    ) -> str:
        return JUDGE_CRITIQUE_USER_PROMPT_TEMPLATE.format(
            user_query=user_query, conversation=conversation_trace, tool_calls_trace=tool_calls_trace
        )


class ExtractScorer(Scorer[ExtractInput]):
    """Extract scalar from assistant answer and compare to expected."""

    def __init__(self, llm_provider: LlmProvider, system_prompt: Optional[str] = None):
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt

    async def score(self, inputs: ExtractInput) -> dict[str, Any]:
        # Get user question from conversation
        user_messages = inputs.conversation.get_user_messages()
        if not user_messages:
            return {
                "extracted": None,
                "expected": inputs.expected,
                "match": False,
                "score": 0.0,
                "mode": EvaluationMode.EXTRACT.value,
                "error": "No user question found",
            }
        question = user_messages[0].content

        # Get assistant answer from conversation
        assistant_messages = inputs.conversation.get_assistant_messages()
        if not assistant_messages:
            return {
                "extracted": None,
                "expected": inputs.expected,
                "match": False,
                "score": 0.0,
                "mode": EvaluationMode.EXTRACT.value,
                "error": "No assistant answer found",
            }
        answer = "\n".join(msg.content for msg in assistant_messages)

        try:
            extracted_str = await self._extract_scalar(question, answer)
            extracted = self._parse_to_type(extracted_str, inputs.expected)
            match = self._values_equal(extracted, inputs.expected)
            return {
                "extracted": extracted,
                "expected": inputs.expected,
                "match": match,
                "score": 1.0 if match else 0.0,
                "mode": EvaluationMode.EXTRACT.value,
            }
        except Exception as e:
            return {
                "extracted": None,
                "expected": inputs.expected,
                "match": False,
                "score": 0.0,
                "mode": EvaluationMode.EXTRACT.value,
                "error": str(e),
            }

    async def _extract_scalar(self, question: str, answer: str) -> str:
        prompt = JUDGE_EXTRACT_USER_PROMPT_TEMPLATE.format(question=question, answer=answer)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await self.llm_provider.chat_complete(messages=messages)
        if not resp.content:
            raise ValueError("Empty LLM result")
        return resp.content.strip()

    def _parse_to_type(self, s: str, expected: Any) -> Any:
        if isinstance(expected, (int, float)):
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return s
        return s

    def _values_equal(self, a: Any, b: Any) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-3
        return str(a).lower() == str(b).lower()


# === Modes and Judge ===
class EvaluationMode(Enum):
    MATCH = "match"
    CRITIQUE = "critique"
    EXTRACT = "extract"


class Judge:
    def __init__(
        self,
        llm_provider: LlmProvider | None = None,
        default_mode: EvaluationMode = EvaluationMode.MATCH,
        system_prompt: Optional[str] = None,
    ):
        self.llm_provider = llm_provider or create_provider()
        self.default_mode = default_mode
        self.system_prompt = system_prompt
        self.context_compressor = ContextCompressor(self.llm_provider)

    async def score(
        self,
        expected: Optional[dict[str, Any]] = None,
        conversation: Optional[ChatHistory] = None,
        tool_calls: Optional[List[dict[str, Any]]] = None,
        mode: Optional[Union[EvaluationMode, Sequence[EvaluationMode]]] = None,
    ) -> dict[str, Any]:
        modes = [self.default_mode] if mode is None else ([mode] if isinstance(mode, EvaluationMode) else list(mode))

        if len(modes) == 1:
            return await self._score_single(modes[0], expected, conversation, tool_calls)

        results: dict[str, Any] = {}
        for m in modes:
            results[m.value] = await self._score_single(m, expected, conversation, tool_calls)
        return results

    async def _score_single(
        self,
        eval_mode: EvaluationMode,
        expected: Optional[dict[str, Any]],
        conversation: Optional[ChatHistory],
        tool_calls: Optional[List[dict[str, Any]]],
    ) -> dict[str, Any]:
        if eval_mode == EvaluationMode.MATCH:
            if not conversation or expected is None:
                raise ValueError("conversation and expected are required for MATCH")
            assistant_messages = conversation.get_assistant_messages()
            if not assistant_messages:
                raise ValueError("No assistant messages found in conversation")
            return await MatchScorer().score(MatchInput(assistant_messages[-1].content, expected))

        if eval_mode == EvaluationMode.CRITIQUE:
            if not conversation:
                raise ValueError("conversation is required for CRITIQUE")
            return await CritiqueScorer(self.llm_provider, self.context_compressor, self.system_prompt).score(
                CritiqueInput(conversation, tool_calls)
            )

        if eval_mode == EvaluationMode.EXTRACT:
            if not conversation or expected is None:
                raise ValueError("conversation and expected are required for EXTRACT")
            return await ExtractScorer(self.llm_provider, self.system_prompt).score(
                ExtractInput(conversation, expected)
            )

        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
