from __future__ import annotations
from typing import Any

from lev.core.llm_provider import LlmProvider
from lev.prompts.judge import JUDGE_EXTRACT_USER_PROMPT_TEMPLATE
from lev.scoring import Score, Scorer, ScoringContext

class LLMExtractValueScorer(Scorer):
    """Extract scalar from assistant answer and compare to expected."""

    def __init__(self, llm_provider: LlmProvider, system_prompt: str | None = None, expected: Any = None, **kwargs):
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt
        self.expected = expected

    @property
    def display_name(self) -> str:
        return "LLM Extract Value Scorer"

    async def score(self, ctx: ScoringContext) -> Score:
        conversation = ctx.chat_history
        expected = self.expected

        if expected is None:
            return Score(0.0, "No expected value provided")

        # Get user question from conversation
        user_messages = conversation.get_user_messages()
        if not user_messages:
            return Score(0.0, "No user question found")
        question = user_messages[0].content

        # Get assistant answer from conversation
        assistant_messages = conversation.get_assistant_messages()
        if not assistant_messages:
            return Score(0.0, "No assistant answer found")
        answer = "\n".join(msg.content for msg in assistant_messages)

        try:
            extracted_str = await self._extract_scalar(question, answer)
            extracted = self._parse_to_type(extracted_str, expected)
            match = self._values_equal(extracted, expected)
            
            score_value = 1.0 if match else 0.0
            reason = f"Expected: {expected}, Extracted: {extracted}, Match: {match}"
            
            return Score(score_value, reason)
        except Exception as e:
            return Score(0.0, f"Extraction failed: {e}")

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

def create_llm_extract_value_scorer(llm_provider: LlmProvider, system_prompt: str | None = None, expected: Any = None, **kwargs) -> LLMExtractValueScorer:
    """Factory function to create LLMExtractValueScorer."""
    return LLMExtractValueScorer(llm_provider, system_prompt, expected, **kwargs)
