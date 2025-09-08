from lev.judge import Judge, EvaluationMode
from lev.scoring import Score, Scorer, ScoringContext


class LLMJudgeScorer(Scorer):
    """Scorer that uses the existing Judge with LLM evaluation."""

    def __init__(self, judge: Judge, mode: EvaluationMode = EvaluationMode.CRITIQUE, expected=None):
        self.judge = judge
        self.mode = mode
        self.expected = expected

    @property
    def display_name(self) -> str:
        if self.mode == EvaluationMode.EXTRACT:
            return "LLM Judge (Extraction)"
        elif self.mode == EvaluationMode.MATCH:
            return "LLM Judge (Matching)"
        elif self.mode == EvaluationMode.CRITIQUE:
            return "LLM Judge (Critique)"
        return "LLM Judge"

    async def score(self, ctx: ScoringContext) -> Score:
        """Score using LLM judge."""
        try:
            # For EXTRACT mode, use the expected value from parameters if provided
            expected_value = (
                self.expected if self.mode == EvaluationMode.EXTRACT and self.expected is not None else ctx.expected
            )

            result = await self.judge.score(
                expected=expected_value, conversation=ctx.chat_history, tool_calls=ctx.tool_calls, mode=self.mode
            )

            # Extract score and reasoning based on mode
            if self.mode == EvaluationMode.EXTRACT:
                score_value = result.get("score", 0.0)
                extracted = result.get("extracted")
                expected = result.get("expected")
                match = result.get("match", False)
                error = result.get("error")

                if error:
                    reason = f"Extraction failed: {error}"
                else:
                    reason = f"Extracted: '{extracted}', expected: '{expected}', match: {match}"
            else:
                score_value = result.get("score", 0.0)
                reason = result.get("justification", result.get("reasoning", "No reasoning provided"))

            return Score(score_value, reason)

        except Exception as e:
            return Score(0.0, f"LLM Judge evaluation failed: {str(e)}")


def create_llm_judge_scorer(judge: Judge, mode: str = "critique", **kwargs) -> LLMJudgeScorer:
    """Factory method to create an LLM judge scorer."""
    eval_mode = EvaluationMode.CRITIQUE
    if mode == "extract":
        eval_mode = EvaluationMode.EXTRACT
    elif mode == "match":
        eval_mode = EvaluationMode.MATCH
    elif mode == "critique":
        eval_mode = EvaluationMode.CRITIQUE

    # Extract expected value for EXTRACT mode
    expected = kwargs.get("expected") if eval_mode == EvaluationMode.EXTRACT else None

    return LLMJudgeScorer(judge, eval_mode, expected)
