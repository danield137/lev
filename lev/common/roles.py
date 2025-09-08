from enum import Enum


class MessageRole(str, Enum):
    """Standard message roles for chat conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"
    PLATFORM = "platform"
