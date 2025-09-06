from dataclasses import dataclass
from enum import Enum
from typing import Any

from lev.core.config import Eval
from lev.core.mcp import McpClientRegistry, McpServerConfig
from lev.core.provider_registry import LlmProviderRegistry
from lev.core.results import ResultSink
from lev.llm_config_loader import LLMConfig


@dataclass(slots=True)
class ResolvedEvalManifest:
    name: str
    provider_registry: LlmProviderRegistry
    mcps: dict[str, McpServerConfig]
    mcp_registry: McpClientRegistry
    evals: list[Eval]
    result_sink: ResultSink | None = None


class DatasetType(str, Enum):
    MCP_EVAL = "mcp_eval"


@dataclass(slots=True)
class EvalManifest:
    schema_version: str
    type: DatasetType
    description: str
    llm_config: LLMConfig
    mcp_servers: dict[str, McpServerConfig]
    evals: list[Eval]
    logging: dict[str, Any] | None = None
