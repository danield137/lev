"""
Base agent abstract class.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from lev.core.chat_history import ChatHistory
from lev.core.llm_provider import LlmProvider


class Agent(ABC):
    chat_history: ChatHistory
    llm_provider: LlmProvider
    system_prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    def __init__(
        self,
        llm_provider: LlmProvider,
        system_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        self.llm_provider = llm_provider
        self.chat_history = ChatHistory()
        self.system_prompt = system_prompt
        self.chat_history.add_system_message(system_prompt)
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def initialize(self) -> None:
        """Initialize the agent by connecting to MCP servers."""
        ...

    async def cleanup(self) -> None:
        """Clean up MCP connections."""
        ...

    @abstractmethod
    async def message(self, user_message: str, track: bool = True) -> str | None:
        """Process a user message and return a response."""
        ...

    async def reset(self) -> None:
        """Reset the agent's chat history."""
        self.chat_history = ChatHistory()
        self.chat_history.add_system_message(self.system_prompt)


class SimpleAgent(Agent):
    async def message(self, user_message: str, track: bool = True) -> str | None:
        if track:
            self.chat_history.add_user_message(user_message)
        messages = [{"role": m.role, "content": m.content} for m in self.chat_history.get_conversation(with_system=True)]  # type: ignore
        response = await self.llm_provider.chat_complete(messages)
        result = None
        if response and response.content:
            result = response.content.strip()
            if track:
                self.chat_history.add_assistant_message(result)
        return result
