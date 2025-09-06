from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from lev.core.agent import Agent
from lev.core.chat_history import ChatHistory


@dataclass(slots=True)
class ConversationResult:
    conversation: ChatHistory
    mcps: List[str]
    success: bool
    error: Optional[str] = None
    solver_agent: Optional[Agent] = None


@dataclass(slots=True)
class McpEvaluationResult:
    eval_id: str
    question: str
    score: float
    reasoning: str
    conversation: ChatHistory
    mcps: List[str]
    mcp_valid: bool
    tool_calls_sequence: List[Dict[str, Any]]
    conversation_trace: Optional[str] = ""
    individual_scores: Optional[Dict[str, float]] = None


class ResultSink(Protocol):
    """Protocol for outputting evaluation results to various destinations."""

    def write(self, results: list[McpEvaluationResult]) -> None:
        """Write evaluation results to the sink destination."""
        ...
