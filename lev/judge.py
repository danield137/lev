from enum import Enum
from typing import Any, Sequence

from lev.context_compressor import ContextCompressor
from lev.core.chat_history import ChatHistory
from lev.core.llm_provider import LlmProvider
from lev.llm_providers import create_provider
from lev.scoring import ScoringContext
from lev.scoring.llm.critique import LLMCritiqueScorer
from lev.scoring.llm.extract_value import LLMExtractValueScorer

# === Modes and Judge ===
class EvaluationMode(Enum):
    CRITIQUE = "critique"
    EXTRACT = "extract"

class Judge:
    def __init__(
        self,
        llm_provider: LlmProvider | None = None,
        default_mode: EvaluationMode = EvaluationMode.CRITIQUE,
        system_prompt: str | None = None,
    ):
        self.llm_provider = llm_provider or create_provider()
        self.default_mode = default_mode
        self.system_prompt = system_prompt
        self.context_compressor = ContextCompressor(self.llm_provider)

    async def score(
        self,
        expected: Any = None,
        conversation: ChatHistory | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        mode: EvaluationMode | Sequence[EvaluationMode] | None = None,
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
        expected: Any,
        conversation: ChatHistory | None,
        tool_calls: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if not conversation:
            raise ValueError("conversation is required")

        # Create scoring context
        ctx = ScoringContext(
            chat_history=conversation,
            tool_calls=tool_calls,
            expected=expected
        )

        if eval_mode == EvaluationMode.CRITIQUE:
            scorer = LLMCritiqueScorer(self.llm_provider, self.context_compressor, self.system_prompt)
            score_result = await scorer.score(ctx)
            return {
                "score": score_result.value,
                "justification": score_result.reason,
                "mode": "critique"
            }

        if eval_mode == EvaluationMode.EXTRACT:
            if expected is None:
                raise ValueError("expected is required for EXTRACT mode")
            scorer = LLMExtractValueScorer(self.llm_provider, self.system_prompt)
            score_result = await scorer.score(ctx)
            return {
                "score": score_result.value,
                "justification": score_result.reason,
                "mode": "extract"
            }

        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
