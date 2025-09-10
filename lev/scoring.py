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

        # Filter out scorers with 0 weights to avoid calling them
        active_scorers = [(w, s) for w, s in self.weighted_scorers if w > 0]
        
        if not active_scorers:
            return Score(0.0, "No active scorers (all weights are 0)")

        subtotal = 0.0
        total_weight = sum(w for w, _ in active_scorers)
        reasons = []

        for weight, scorer in active_scorers:
            score_result = await scorer.score(ctx)
            weighted_value = weight * score_result.value
            subtotal += weighted_value

            scorer_name = scorer.__class__.__name__
            reasons.append(f"{scorer_name}={score_result.value:.2f} ({score_result.reason}) *{weight}")

        overall_score = subtotal / total_weight if total_weight > 0 else 0.0
        combined_reason = "; ".join(reasons)

        return Score(overall_score, combined_reason)




def create_contains_string_scorer(target_string: str, case_sensitive: bool = False) -> ContainsStringScorer:
    """Factory method to create a contains string scorer."""
    return ContainsStringScorer(target_string, case_sensitive)




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
