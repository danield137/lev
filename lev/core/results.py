from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    suite_id: str
    question: str
    score: float
    reasoning: str
    conversation: ChatHistory
    mcps: List[str]
    mcp_valid: bool
    tool_calls_sequence: List[Dict[str, Any]]
    conversation_trace: Optional[str] = ""
    individual_scores: Optional[Dict[str, float]] = None
