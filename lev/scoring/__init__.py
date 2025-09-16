from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from termcolor import colored

from lev.core.chat_history import ChatHistory


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

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return a display name for the scorer."""
        pass

    @abstractmethod
    async def score(self, ctx: ScoringContext) -> Score:
        """Score the evaluation context and return a Score."""
        pass


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

        # Filter out zero-weight scorers to avoid unnecessary computation
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

            scorer_name = scorer.display_name
            reasons.append(f"{scorer_name}:{colored(score_result.value, 'green')} ({score_result.reason}) ")

        overall_score = subtotal / total_weight if total_weight > 0 else 0.0
        combined_reason = "\n".join(reasons)

        return Score(overall_score, combined_reason)


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
