from termcolor import colored

from lev.agents.factory import create_tool_agent_from_provider, create_agent_from_provider
from lev.config import Eval
from lev.core import agent
from lev.core.agent import Agent
from lev.core.provider_registry import LlmProviderRegistry
from lev.host.mcp_host import McpHost
from lev.host.mcp_registry import McpClientRegistry
from lev.prompts.introspection import INTROSPECTIVE_AGENT_SYSTEM_PROMPT

def print_header(
    dataset_name: str,
    evals: list[Eval],
    mcp_registry: McpClientRegistry,
    provider_registry: LlmProviderRegistry,
) -> None:
    print("🧪 MCP Host Evaluation Suite")
    print(f"Manifest: {colored(dataset_name, 'magenta')} ({len(evals)} evals)")
    print(
        f"MCP Servers: [{', '.join(colored(name, 'magenta') for name in mcp_registry.list_servers()) or colored('None', 'red')}]"
    )
    # Display active provider information
    providers_info = provider_registry.get_active_providers_info()
    print("Active Providers:")
    for role, info in providers_info.items():
        provider_name = colored(info["name"], "magenta")
        model_name = colored(info["model"], "magenta")
        print(f"  {role}: {provider_name} ({model_name})")

    print("=" * 80)
    print()

async def run_host_evals(
    dataset_name: str,
    evals: list[Eval],
    provider_registry: LlmProviderRegistry,
    mcp_registry: McpClientRegistry,
    limit: int | None = None,
) -> list[dict]:
    # Apply limit if specified
    if limit is not None and limit > 0:
        evals = evals[:limit]

    print_header(dataset_name, evals, mcp_registry, provider_registry)

    # Create solver agent from provider registry
    # TODO: fix this later. old runner code recreated the agent for each eval, but with host, we only have one
    solver_agent = create_tool_agent_from_provider(evals[0], provider_registry.get_solver(), mcp_registry)
    # TODO: hacky way to create an introspector
    introspector = create_agent_from_provider(evals[0], provider_registry.get_solver())
    introspector.system_prompt = INTROSPECTIVE_AGENT_SYSTEM_PROMPT
    # Create McpHost with solver agent and MCP registry
    host = McpHost(
        agent=solver_agent,
        mcp_registry=mcp_registry,
        introspector=introspector
    )

    results = []
    
    try:
        for i, eval_item in enumerate(evals, 1):
            eval_id = eval_item.id
            question = eval_item.question
            
            print(f"\n[{i}/{len(evals)}] Eval: {colored(eval_id, 'cyan')}")
            print(f"Question: {question}")
            print("-" * 40)

            try:
                # Reset host for each new evaluation
                await host.reset()
                
                # Get answer from host
                answer = await host.prompt(question)
                
                print(f"Answer: {answer}")
                print(f"Journal entries: {len(host.journal)}")
                
                # TODO: Add evaluation logic here
                # evaluator = Evaluator(..)
                # score = evaluator.score(answer, eval_item.expected, eval_item.scoring)
                # print(f"Score: {score}")
                
                # Store result
                result = {
                    "eval_id": eval_id,
                    "question": question,
                    "answer": answer,
                    "journal_entries": len(host.journal),
                    "success": True,
                    # "score": score,  # TODO: Add when evaluator is implemented
                }
                
            except Exception as e:
                print(f"❌ Error in eval {eval_id}: {e}")
                result = {
                    "eval_id": eval_id,
                    "question": question,
                    "answer": f"Error: {e}",
                    "journal_entries": 0,
                    "success": False,
                    # "score": 0.0,  # TODO: Add when evaluator is implemented
                }

            results.append(result)
            
            # TODO: Add summary update here
            # summary.add(eval_item, result["answer"], result.get("score", 0.0))
            print("-" * 40)

    finally:
        # Cleanup
        await host.cleanup()

    print(f"\nCompleted {len(results)} evaluations for {dataset_name}")
    return results
