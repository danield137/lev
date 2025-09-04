from typing import Any, Dict, List, Optional

from termcolor import colored

from lev.conversation import converse
from lev.core.chat_history import ChatHistory
from lev.core.mcp import McpClientRegistry
from lev.core.provider_registry import LlmProviderRegistry
from lev.core.results import McpEvaluationResult
from lev.dataset_loader import ModelConfig, validate_mcp_usage
from lev.judge import EvaluationMode, Judge
from lev.llm_providers.provider_factory import create_provider
from lev.reporting import print_suite_result, print_summary


def print_header(
    dataset_name: str,
    suites: List[Dict[str, Any]],
    mcp_registry: McpClientRegistry,
    provider_registry: LlmProviderRegistry,
) -> None:
    print("🧪 MCP Evaluation Suite")
    print(f"Dataset: {colored(dataset_name, 'magenta')} ({len(suites)} suites)")
    print(
        f"MCP Servers: [{', '.join(colored(name, 'magenta') for name in mcp_registry.list_servers()) or colored('None', 'red')}]"
    )
    print(f"Available Roles: [{', '.join(colored(role, 'magenta') for role in provider_registry.roles())}]")

    # Display active provider information
    providers_info = provider_registry.get_active_providers_info()
    print("Active Providers:")
    for role, info in providers_info.items():
        provider_name = colored(info["name"], "magenta")
        model_name = colored(info["model"], "magenta")
        print(f"  {role}: {provider_name} ({model_name})")

    print("=" * 80)
    print()


def create_judge(model_config: Optional[ModelConfig] = None) -> Judge:
    """Create a Judge instance based on suite configuration."""
    if model_config:
        judge_provider = create_provider(
            provider_name=model_config.provider, model=model_config.model, **model_config.model_parameters
        )
    else:
        judge_provider = create_provider()
    return Judge(judge_provider)


async def run_evals(
    dataset_name: str,
    suites: List[Dict[str, Any]],
    provider_registry: LlmProviderRegistry,
    mcp_registry: McpClientRegistry,
    limit: Optional[int] = None,
) -> List[McpEvaluationResult]:
    # Apply limit if specified
    if limit is not None and limit > 0:
        suites = suites[:limit]

    print_header(dataset_name, suites, mcp_registry, provider_registry)

    # Create judge from provider registry
    judge = Judge(provider_registry.get_judge())

    results = []
    display_names = []
    for i, suite in enumerate(suites, 1):
        suite_id = suite["id"]
        question = suite["question"]
        suite_mcps = suite["mcps"]
        tool_calls_sequence = []

        # Create display name with tags
        tags = suite.get("tags", [])
        display_id = f"{suite_id} [{', '.join(tags)}]" if tags else suite_id
        display_names.append(display_id)

        try:
            # Run conversation simulation
            conversation_result = await converse(suite, mcp_registry, provider_registry)

            if not conversation_result.success:
                print(f"❌ Error in suite {display_id}: {conversation_result.error}")
                results.append(
                    McpEvaluationResult(
                        suite_id=suite_id,
                        question=question,
                        score=0.0,
                        reasoning=conversation_result.error or "Unknown error",
                        conversation=ChatHistory(),
                        mcps=[],
                        mcp_valid=False,
                        tool_calls_sequence=[],
                    )
                )
                print_suite_result(results[-1], i, len(suites), display_id)
                continue

            conversation = conversation_result.conversation
            mcps = conversation_result.mcps
            solver_agent = conversation_result.solver_agent
            if solver_agent is None:
                raise ValueError("No solver agent found.")

            for msg in solver_agent.chat_history.messages:
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tool_calls_sequence.append(
                            {
                                "type": "call",
                                "name": tc["function"]["name"],
                                "server": mcps[0] if mcps else "unknown",
                                "arguments": tc["function"]["arguments"],
                            }
                        )
                elif msg.get("role") == "tool":
                    tool_calls_sequence.append({"type": "response", "content": msg.get("content", "")})

            # Validate MCP usage
            mcp_valid = validate_mcp_usage(suite, mcps)

            # Judge the conversation using scoring configuration
            if len(conversation) >= 2:
                try:
                    conversation_trace = ""
                    if solver_agent and getattr(solver_agent, "chat_history", None):
                        conversation_trace = solver_agent.chat_history.render_trace()

                    # Handle scoring configuration from suite
                    scoring_config = suite.get("scoring", ["critique"])
                    scores = []
                    all_reasoning = []

                    for scoring_method in scoring_config:
                        if isinstance(scoring_method, str):
                            # Simple string mode like "critique"
                            mode = EvaluationMode.CRITIQUE if scoring_method == "critique" else EvaluationMode.MATCH
                            score_result = await judge.score(
                                conversation=conversation,
                                tool_calls=tool_calls_sequence,
                                mode=mode,
                            )
                        elif isinstance(scoring_method, dict) and scoring_method.get("type") == "llm_judge":
                            # Dict-based scoring like {"type": "llm_judge", "mode": "extract", "expected": 15}
                            mode_str = scoring_method.get("mode", "critique")
                            if mode_str == "extract":
                                mode = EvaluationMode.EXTRACT
                                expected = scoring_method.get("expected")
                                score_result = await judge.score(
                                    conversation=conversation,
                                    mode=mode,
                                    expected=expected,
                                )
                            elif mode_str == "match":
                                mode = EvaluationMode.MATCH
                                expected = scoring_method.get("expected")
                                score_result = await judge.score(
                                    conversation=conversation,
                                    mode=mode,
                                    expected=expected,
                                )
                            else:
                                mode = EvaluationMode.CRITIQUE
                                score_result = await judge.score(
                                    conversation=conversation,
                                    mode=mode,
                                )
                        else:
                            continue

                        scores.append(score_result.get("score", 0.0))

                        # Extract reasoning based on the scoring method type
                        if isinstance(scoring_method, dict) and scoring_method.get("mode") == "extract":
                            # For EXTRACT mode, create reasoning from the extraction results
                            extracted = score_result.get("extracted")
                            expected = score_result.get("expected")
                            match = score_result.get("match", False)
                            error = score_result.get("error")

                            if error:
                                reasoning_text = f"Extraction failed: {error}"
                            else:
                                reasoning_text = f"Extracted:'{extracted}', expected:'{expected}', match:'{match}'"
                        else:
                            # For other modes, use justification or reasoning fields
                            reasoning_text = score_result.get(
                                "justification", score_result.get("reasoning", "No reasoning provided")
                            )

                        # Format the scoring method name consistently
                        if isinstance(scoring_method, dict) and scoring_method.get("type") == "llm_judge":
                            mode_name = scoring_method.get("mode", "unknown")
                            method_label = f"llm_judge.{mode_name}"
                        else:
                            method_label = str(scoring_method)

                        all_reasoning.append(f"{method_label}: {reasoning_text}")

                    # Combine scores (average for now)
                    score = sum(scores) / len(scores) if scores else 0.0
                    reasoning = "\n".join(all_reasoning)

                    # Store individual scores for breakdown display
                    individual_scores = {}
                    for idx, scoring_method in enumerate(scoring_config):
                        if isinstance(scoring_method, str):
                            individual_scores[scoring_method] = scores[idx] if idx < len(scores) else 0.0
                        elif isinstance(scoring_method, dict) and scoring_method.get("type") == "llm_judge":
                            mode_name = scoring_method.get("mode", "unknown")
                            individual_scores[mode_name] = scores[idx] if idx < len(scores) else 0.0

                except Exception as e:
                    score = 0.0
                    reasoning = f"Judge evaluation failed: {str(e)}"
                    conversation_trace = ""
            else:
                score = 0.0
                reasoning = "No conversation to evaluate"
                conversation_trace = ""

            # Penalize if wrong MCPs were used
            if not mcp_valid:
                score *= 0.5
                reasoning += " (Score reduced due to invalid MCP usage)"

            results.append(
                McpEvaluationResult(
                    suite_id=suite_id,
                    question=question,
                    score=score,
                    reasoning=reasoning,
                    conversation=conversation,
                    mcps=mcps,
                    mcp_valid=mcp_valid,
                    tool_calls_sequence=tool_calls_sequence,
                    conversation_trace=conversation_trace,
                    individual_scores=individual_scores if "individual_scores" in locals() else {},  # type: ignore
                )
            )
            print_suite_result(results[-1], i, len(suites), display_id)

        except Exception as e:
            print(f"❌ Error in suite {display_id}: {e}")
            results.append(
                McpEvaluationResult(
                    suite_id=suite_id,
                    question=question,
                    score=0.0,
                    reasoning=str(e),
                    conversation=ChatHistory(),
                    mcps=[],
                    mcp_valid=False,
                    tool_calls_sequence=[],
                    conversation_trace="",
                )
            )
            print_suite_result(results[-1], i, len(suites), display_id)

    # Print final summary
    print_summary(results, final=True, display_names=display_names)

    return results
