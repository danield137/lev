"""
LLM Provider Registry for role-based provider access with fallbacks.
"""

from dataclasses import dataclass
from typing import Dict, List

from lev.core.llm_provider import LlmProvider


@dataclass(slots=True)
class LlmProviderRegistry:
    """Registry for managing LLM providers by role with automatic fallbacks."""

    _providers: Dict[str, LlmProvider]

    def get_solver(self) -> LlmProvider:
        """Get solver provider (required role)."""
        return self._providers["solver"]

    def get_judge(self) -> LlmProvider:
        """Get judge provider, fallback to solver if not provided."""
        return self._providers.get("judge", self.get_solver())

    def get_asker(self) -> LlmProvider:
        """Get asker provider, fallback to solver if not provided."""
        return self._providers.get("asker", self.get_solver())

    def get(self, role: str) -> LlmProvider:
        """Get provider for specific role, fallback to solver if not found."""
        if role in self._providers:
            return self._providers[role]
        # Ultimate fallback to solver
        return self.get_solver()

    def roles(self) -> List[str]:
        """Get list of available provider roles."""
        return list(self._providers.keys())

    def has_role(self, role: str) -> bool:
        """Check if a specific role is explicitly defined."""
        return role in self._providers

    def get_active_providers_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about all active providers."""
        info = {}
        for role, provider in self._providers.items():
            info[role] = {"name": provider.name, "model": provider.default_model or "N/A"}
        return info
