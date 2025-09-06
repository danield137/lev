from lev.agents.reasoning import ReasoningAgent
from lev.core.agent import SimpleAgent
from lev.core.config import Eval
from lev.core.llm_provider import LlmProvider
from lev.core.mcp import McpClientRegistry
from lev.loader import get_persona_system_prompt
from lev.prompts.reasoning import REASONING_AGENT_DEFAULT_SYSTEM_PROMPT


def create_mcp_clients(eval: Eval, mcp_registry: McpClientRegistry):
    """Create MCP clients for allowed servers using the registry."""
    mcps = eval.execution.mcps
    mcp_clients = []
    for server_name in mcps:
        mcp_client = mcp_registry.get_client(server_name)
        if mcp_client:
            mcp_clients.append(mcp_client)
    return mcp_clients


def create_agent_from_provider(eval: Eval, provider: LlmProvider) -> SimpleAgent:
    """Create an asker agent using a provider directly."""
    asker_persona = "diligent_asker"
    asker_prompt = get_persona_system_prompt(asker_persona)
    return SimpleAgent(llm_provider=provider, system_prompt=asker_prompt)


def create_reasoning_agent_from_provider(
    eval: Eval, provider: LlmProvider, mcp_registry: McpClientRegistry | None = None
) -> ReasoningAgent:
    """Create a reasoning agent using a provider directly."""
    mcp_clients = []
    if mcp_registry:
        mcp_clients = create_mcp_clients(eval, mcp_registry)
    return ReasoningAgent(
        llm_provider=provider, system_prompt=REASONING_AGENT_DEFAULT_SYSTEM_PROMPT, mcp_clients=mcp_clients
    )
