from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from lev.core.chat_history import ChatHistory
from lev.core.llm_provider import LlmProvider
from lev.judge import Judge, EvaluationMode


@dataclass(slots=True)
class Score:
    """Result of a scoring operation."""

    value: float
    reason: str


@dataclass(slots=True)
class ScoringContext:
    """Context passed to scorers containing all evaluation data."""

    chat_history: ChatHistory
    answer: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    expected: dict[str, Any] | None = None


class Scorer(ABC):
    """Base interface for scoring implementations."""

    @abstractmethod
    async def score(self, ctx: ScoringContext) -> Score:
        """Score the evaluation context and return a Score."""
        pass


class LLMJudgeScorer(Scorer):
    """Scorer that uses the existing Judge with LLM evaluation."""

    def __init__(self, judge: Judge, mode: EvaluationMode = EvaluationMode.CRITIQUE):
        self.judge = judge
        self.mode = mode

    async def score(self, ctx: ScoringContext) -> Score:
        """Score using LLM judge."""
        try:
            result = await self.judge.score(
                expected=ctx.expected, conversation=ctx.chat_history, tool_calls=ctx.tool_calls, mode=self.mode
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


class ContainsStringScorer(Scorer):
    """Static scorer that checks if answer contains a specific string."""

    def __init__(self, target_string: str, case_sensitive: bool = False):
        self.target_string = target_string
        self.case_sensitive = case_sensitive

    async def score(self, ctx: ScoringContext) -> Score:
        """Score based on whether the answer contains the target string."""
        if not ctx.answer:
            return Score(0.0, f"No answer to check for '{self.target_string}'")

        search_text = ctx.answer if self.case_sensitive else ctx.answer.lower()
        target = self.target_string if self.case_sensitive else self.target_string.lower()

        if target in search_text:
            return Score(1.0, f"Found '{self.target_string}' in answer")
        else:
            return Score(0.0, f"'{self.target_string}' not found in answer")


class ScoreFunction:
    """Orchestrates multiple scorers with weights to produce an overall score."""

    def __init__(self, weighted_scorers: list[tuple[float, Scorer]]):
        """
        Initialize with weighted scorers.

        Args:
            weighted_scorers: List of (weight, scorer) tuples
        """
        self.weighted_scorers = weighted_scorers

    async def score(self, ctx: ScoringContext) -> Score:
        """Execute all scorers and return weighted aggregate score."""
        if not self.weighted_scorers:
            return Score(0.0, "No scorers configured")

        subtotal = 0.0
        total_weight = sum(w for w, _ in self.weighted_scorers)
        reasons = []

        for weight, scorer in self.weighted_scorers:
            score_result = await scorer.score(ctx)
            weighted_value = weight * score_result.value
            subtotal += weighted_value

            scorer_name = scorer.__class__.__name__
            reasons.append(f"{scorer_name}={score_result.value:.2f} ({score_result.reason}) *{weight}")

        overall_score = subtotal / total_weight if total_weight > 0 else 0.0
        combined_reason = "; ".join(reasons)

        return Score(overall_score, combined_reason)


def create_llm_judge_scorer(judge: Judge, mode: str = "critique") -> LLMJudgeScorer:
    """Factory method to create an LLM judge scorer."""
    eval_mode = EvaluationMode.CRITIQUE
    if mode == "extract":
        eval_mode = EvaluationMode.EXTRACT
    elif mode == "match":
        eval_mode = EvaluationMode.MATCH
    elif mode == "critique":
        eval_mode = EvaluationMode.CRITIQUE

    return LLMJudgeScorer(judge, eval_mode)


def create_contains_string_scorer(target_string: str, case_sensitive: bool = False) -> ContainsStringScorer:
    """Factory method to create a contains string scorer."""
    return ContainsStringScorer(target_string, case_sensitive)


def build_scorers(scoring_config: list[dict[str, Any]], judge: Judge) -> list[tuple[float, Scorer]]:
    """
    Build a list of weighted scorers from configuration.

    Args:
        scoring_config: List of scorer configuration dictionaries
        judge: Judge instance for LLM-based scoring

    Returns:
        List of (weight, scorer) tuples
    """
    weighted_scorers = []

    for config in scoring_config:
        if isinstance(config, str):
            # Simple string config like "critique"
            weight = 1.0
            if config == "critique":
                scorer = create_llm_judge_scorer(judge, "critique")
            elif config == "match":
                scorer = create_llm_judge_scorer(judge, "match")
            else:
                continue  # Skip unknown string configs
        elif isinstance(config, dict):
            # Dictionary config
            scorer_type = config.get("type")
            weight = config.get("weight", 1.0)

            if scorer_type == "llm_judge":
                mode_str = config.get("mode", "critique")
                scorer = create_llm_judge_scorer(judge, mode_str)

            elif scorer_type == "contains_string":
                target_string = config.get("value")
                if not target_string:
                    continue  # Skip if no target string

                case_sensitive = config.get("case_sensitive", False)
                scorer = create_contains_string_scorer(target_string, case_sensitive)
            else:
                continue  # Skip unknown scorer types
        else:
            continue  # Skip invalid config

        weighted_scorers.append((weight, scorer))

    return weighted_scorers


def validate_mcp_usage(eval_mcps: list[str], used_mcps: list[str]) -> bool:
    """
    Validate that only allowed MCPs were used in an evaluation.

    Args:
        eval_mcps: List of MCP server names allowed by the eval
        used_mcps: List of MCP server names that were actually used

    Returns:
        True if all used MCPs are allowed, False otherwise
    """
    allowed = set(eval_mcps)
    used = set(used_mcps)

    # Check if any disallowed MCPs were used
    disallowed_usage = used - allowed
    return len(disallowed_usage) == 0
