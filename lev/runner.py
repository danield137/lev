from termcolor import colored

from lev.agents.factory import create_introspector_from_provider, create_tool_agent_from_provider
from lev.config import Eval
from lev.workflow import AgentWorkflow
from lev.core.provider_registry import LlmProviderRegistry
from lev.mcp.mcp_host import McpHost
from lev.mcp.mcp_registry import McpClientRegistry
from lev.scoring.evaluation import score_evaluation


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
    # Create introspector using factory function
    introspector = create_introspector_from_provider(evals[0], provider_registry.get_solver())

    # Create McpHost with solver agent and MCP registry (no more introspector param)
    host = McpHost(agent=solver_agent, mcp_registry=mcp_registry)
    await host.warm_up()
    # Create AgentWorkflow with host and introspector
    aw = AgentWorkflow(host, introspector)

    results = []

    try:
        for i, eval_item in enumerate(evals, 1):
            eval_id = eval_item.id
            question = eval_item.question

            print(f"\n[{i}/{len(evals)}] Eval: {colored(eval_id, 'cyan')}")
            print("-" * 20)
            print(f"{colored('Question', 'cyan')}: {question}")

            try:
                # Get answer from workflow
                answer = await aw.ask(question)

                print(f"{colored('Answer', 'cyan')}: {answer}")

                # Score the evaluation
                try:
                    score_result = await score_evaluation(
                        eval_item,
                        host.agent.chat_history,
                        answer,
                        host.agent.chat_history.tool_calls,
                        provider_registry,
                    )

                    print("-" * 20)
                    print(f"{colored('Score', 'cyan')}: {score_result.value:.2f}")
                    print("-" * 20)
                    print(f"{colored('Reasoning', 'cyan')}: \n{score_result.reason}")

                except Exception as scoring_error:
                    print(f"❌ Scoring failed: {scoring_error}")
                    from lev.scoring import Score

                    score_result = Score(0.0, f"Scoring failed: {str(scoring_error)}")

                print("-" * 20)
                print("chat_history")
                print("=" * 20)
                print(host.agent.chat_history.render_trace())

                # Store result
                result = {
                    "eval_id": eval_id,
                    "question": question,
                    "answer": answer,
                    "score": score_result.value,
                    "reasoning": score_result.reason,
                    "success": True,
                }

            except Exception as e:
                print(f"❌ Error in eval {eval_id}: {e}")
                result = {
                    "eval_id": eval_id,
                    "question": question,
                    "answer": f"Error: {e}",
                    "score": 0.0,
                    "reasoning": str(e),
                    "success": False,
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
