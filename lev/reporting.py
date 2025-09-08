import json

from termcolor import colored

from lev.results import McpEvaluationResult


def get_result_status(score: float) -> str:
    """Get status string based on score."""
    if score >= 0.9:
        return f"{colored('✅ Success', 'green')}"
    if score > 0.1:
        return f"{colored('⚠️ Partial', 'yellow')}"
    return f"{colored('❌ Failure', 'red')}"


def get_result_icon(score: float) -> str:
    """Get status icon based on score."""
    if score >= 0.9:
        return "✅"
    if score > 0.1:
        return "⚠️"
    return "❌"


def print_suite_result(result: McpEvaluationResult, index: int, total: int, display_name: str | None = None):
    """Prints a formatted result for a single suite."""
    name = display_name if display_name is not None else result.eval_id
    print(f"🎯 Eval {index}/{total}: {name}")
    print("-" * 80)
    print(f"Question : {result.question}")
    print(f"Result   : {get_result_status(result.score)}")
    print(f"Score    : {result.score:.2f}")
    print(f"Used MCPs : {result.mcps}")
    print()
    print("Reasoning")
    print("---------")
    print(result.reasoning)
    print()
    print("Conversation Trace")
    print("------------------")
    if getattr(result, "conversation_trace", None):
        print(result.conversation_trace)
    elif result.conversation:
        role_user = colored("USER", "cyan")
        role_asst = colored("ASSISTANT", "cyan")

        # First user line
        print(f"{role_user}      → {result.conversation[0]['content']}")

        # Assistant trace
        role_prefix_printed = False
        role_prefix = f"{role_asst} → "
        cont = "          "  # fixed 10-space continuation indent to keep alignment consistent

        if result.tool_calls_sequence:
            for item in result.tool_calls_sequence:
                if item["type"] == "call":
                    name = item.get("name", "")
                    server = item.get("server", "unknown")
                    # Determine server name and color it cyan
                    if "." in name:
                        sn, func = name.split(".", 1)
                    else:
                        sn, func = server, name
                    sn_disp = colored(sn, "cyan")
                    full_name = f"{sn_disp}.{func}"

                    try:
                        args_dict = json.loads(item.get("arguments", "") or "{}")
                        args_str = ", ".join([f'{k}="{v}"' for k, v in args_dict.items()])
                    except Exception:
                        args_str = item.get("arguments", "") or ""

                    if not role_prefix_printed:
                        print(f"{role_prefix}{full_name}({args_str})")
                        role_prefix_printed = True
                    else:
                        print(f"{cont}{full_name}({args_str})")

                elif item["type"] == "response":
                    content: str = item.get("content", "")
                    max_length = 100
                    try:
                        content = content.strip()
                        if content.startswith("[") or content.startswith("{"):
                            parsed_response = json.loads(content)
                            if isinstance(parsed_response, dict):
                                if "result" in parsed_response:
                                    content = str(parsed_response["result"])
                                elif "content" in parsed_response:
                                    content = parsed_response["content"]
                                elif "error" in parsed_response:
                                    content = f"Error: {parsed_response['error']}"
                                    max_length = 300  # show more of error messages
                            elif isinstance(parsed_response, list):
                                content = ", ".join(map(str, parsed_response))
                    except Exception:
                        content = content

                    if len(content) > max_length:
                        trimmed = content[:max_length]
                        preview = trimmed + f"... ({len(content.split())-len(trimmed.split())} tokens excluded)"
                    else:
                        preview = content
                    print(f"{cont}← {preview}")

        # Final assistant message bubble
        if len(result.conversation) > 1:
            print(f"{cont}💬 {result.conversation[1]['content']}")

    print("-" * 80)
    print()


def print_summary(results: list[McpEvaluationResult], final: bool = False, display_names: list[str] | None = None):
    """Print evaluation summary statistics."""
    if not results:
        return
    print("📊 MCP Evaluation Summary")
    print("=" * 80)
    avg_score = sum(r.score for r in results) / len(results)
    valid_mcp_count = sum(1 for r in results if r.mcp_valid)
    print(f"Average Score   : {avg_score:.2f}")
    print(f"Scenarios Tested: {len(results)}")
    print(f"Valid MCP Usage : {100*valid_mcp_count/len(results):.0f}% ({valid_mcp_count}/{len(results)})")
    print()
    print("Results")
    print("-------")
    for i, r in enumerate(results):
        icon = get_result_icon(r.score)
        name = display_names[i] if display_names and i < len(display_names) else r.eval_id
        if hasattr(r, "individual_scores") and r.individual_scores:
            # Show score breakdown if multiple scoring methods were used
            breakdown_parts = [f"{method}:{score:.2f}" for method, score in r.individual_scores.items()]
            breakdown = f" ({', '.join(breakdown_parts)})"
            print(f"{name:<30}: {r.score:.2f} {icon}{breakdown}")
        else:
            print(f"{name:<30}: {r.score:.2f} {icon}")
    print()
    if final:
        print("🎉 Evaluation complete.")
