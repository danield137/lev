"""
Centralized prompt templates for lev package.

This package contains all prompt strings used throughout the lev system,
organized by subsystem and following the naming convention:
SUBSYSTEM_TASK_ROLE_TYPE

Where:
- SUBSYSTEM: The package area (JUDGE, TOOL_AGENT, CONTEXT_COMPRESSOR, etc.)
- TASK: The high-level activity (CRITIQUE, EXTRACT, COMPRESS, etc.)
- ROLE: The perspective (USER, ASSISTANT, SYSTEM)
- TYPE: PROMPT for static strings, PROMPT_TEMPLATE for formatted strings
"""

from lev.prompts.context_compressor import (
    CONTEXT_COMPRESSOR_COMPRESS_USER_PROMPT_TEMPLATE,
)
from lev.prompts.judge import (
    JUDGE_CRITIQUE_USER_PROMPT_TEMPLATE,
    JUDGE_EXTRACT_USER_PROMPT_TEMPLATE,
)
from lev.prompts.tool_agent import (
    TOOL_AGENT_DEFAULT_SYSTEM_PROMPT,
)

__all__ = [
    "JUDGE_CRITIQUE_USER_PROMPT_TEMPLATE",
    "JUDGE_EXTRACT_USER_PROMPT_TEMPLATE",
    "CONTEXT_COMPRESSOR_COMPRESS_USER_PROMPT_TEMPLATE",
    "TOOL_AGENT_DEFAULT_SYSTEM_PROMPT",
    "SEMANTIC_AGENT_DEFAULT_SYSTEM_PROMPT",
]
