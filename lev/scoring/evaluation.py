from typing import Any

from lev.config import Eval
from lev.core.chat_history import ChatHistory
from lev.core.provider_registry import LlmProviderRegistry
from lev.judge import Judge
from lev.scoring import Score, ScoringContext, ScoreFunction
from lev.scoring.factory import build_scorers


async def score_evaluation(
    eval_item: Eval,
    chat_history: ChatHistory,
    answer: str,
    tool_calls: list[dict[str, Any]],
    provider_registry: LlmProviderRegistry,
) -> Score:
    """
    Score an evaluation using the configured scoring methods.

    Args:
        eval_item: The evaluation configuration
        chat_history: The conversation history
        answer: The final answer from the agent
        tool_calls: List of tool calls made during evaluation
        provider_registry: Provider registry for creating judge

    Returns:
        Score with value and reasoning
    """
    # Create judge from provider registry
    judge = Judge(provider_registry.get_judge())

    # Build scorers from configuration
    weighted_scorers = build_scorers(eval_item.scoring, judge)
    score_function = ScoreFunction(weighted_scorers)

    # Create scoring context
    ctx = ScoringContext(
        chat_history=chat_history, answer=answer, tool_calls=tool_calls, expected=eval_item.expectations
    )

    # Execute scoring
    return await score_function.score(ctx)
