from lev.core.chat_history import ChatHistory
from lev.core.llm_provider import LlmProvider
from lev.prompts.context_compressor import (
    CONTEXT_COMPRESSOR_COMPRESS_USER_PROMPT2_TEMPLATE,
    CONTEXT_COMPRESSOR_COMPRESS_USER_PROMPT_TEMPLATE,
)


class ContextCompressor:
    llm_provider: LlmProvider

    def __init__(self, llm_provider: LlmProvider):
        self.llm_provider = llm_provider

    async def compress_prompt(self, prompt) -> str:
        prompt = CONTEXT_COMPRESSOR_COMPRESS_USER_PROMPT2_TEMPLATE.format(
            message_sequence=self._format_message_sequence(prompt)
        )

        response = await self.llm_provider.chat_complete(messages=[{"role": "user", "content": prompt}])

        compressed = response.content.strip() if response.content else None
        return compressed if compressed else prompt

    async def compress_chat(self, chat: ChatHistory) -> str:
        conversation_only = chat.get_conversation()
        messages = [f"{msg.role.capitalize()} [{msg.timestamp}]: {msg.content}" for msg in conversation_only]
        return await self._llm_compress(messages)

    async def _llm_compress(self, messages: list[str]) -> str:
        try:
            prompt = CONTEXT_COMPRESSOR_COMPRESS_USER_PROMPT_TEMPLATE.format(
                message_sequence=self._format_message_sequence(messages)
            )

            response = await self.llm_provider.chat_complete(messages=[{"role": "user", "content": prompt}])

            compressed = response.content.strip() if response.content else None
            return compressed if compressed else messages[0]

        except Exception:
            # Fallback to simple concatenation
            return self._fallback_concatenate(messages)

    def _format_message_sequence(self, messages: list[str]) -> str:
        """Format a sequence of messages for prompt display."""
        formatted = []
        for i, msg in enumerate(messages, 1):
            formatted.append(f"{i}. {msg}")

        return "\n".join(formatted)

    def _fallback_concatenate(self, messages: list[str]) -> str:
        """Fallback method for concatenating messages without LLM."""
        if len(messages) == 1:
            return messages[0]

        main_query = messages[0]
        clarifications = " ".join(messages[1:])
        return f"{main_query}\n\nAdditional context: {clarifications}"
