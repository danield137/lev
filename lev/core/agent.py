from abc import ABC, abstractmethod
from typing import Any, Optional

from lev.common.roles import MessageRole
from lev.core.chat_history import ChatHistory
from lev.core.llm_provider import LlmProvider, ModelResponse


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

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        ...

    async def initialize(self) -> None:
        """Initialize the agent by connecting to MCP servers."""
        ...

    async def cleanup(self) -> None:
        """Clean up MCP connections."""
        ...

    @abstractmethod
    async def message(
        self,
        message: str,
        tools: list[dict[str, Any]] | None = None,
        session: bool = True,
        role: MessageRole = MessageRole.USER,
    ) -> ModelResponse:
        """Process a message and return a response."""
        ...

    async def reset(self) -> None:
        """Reset the agent's chat history."""
        self.chat_history = ChatHistory()
        self.chat_history.add_system_message(self.system_prompt)


class SimpleAgent(Agent):
    @property
    def is_initialized(self) -> bool:
        return True

    async def message(
        self,
        message: str,
        tools: list[dict[str, Any]] | None = None,
        session: bool = True,
        role: MessageRole = MessageRole.USER,
    ) -> ModelResponse:
        self.chat_history.add_message(message, role)
        messages = []
        if session:
            messages = [{"role": m.role, "content": m.content} for m in self.chat_history.get_conversation(with_system=True)]  # type: ignore
        else:
            messages = [{"role": role.value, "content": message}]  # type: ignore
        response = await self.llm_provider.chat_complete(messages)
        if response and response.content and session:
            self.chat_history.add_assistant_message(response.content)
        return response
