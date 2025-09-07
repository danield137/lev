from lev.agents.factory import create_agent_from_provider, create_reasoning_agent_from_provider
from lev.config import Eval
from lev.core.chat_history import ChatHistory
from lev.core.provider_registry import LlmProviderRegistry
from lev.host.mcp_registry import McpClientRegistry
from lev.results import ConversationResult


async def converse(
    eval: Eval, mcp_registry: McpClientRegistry, provider_registry: LlmProviderRegistry
) -> ConversationResult:
    # Create agents using provider registry
    asker = create_agent_from_provider(eval, provider_registry.get_asker())
    solver = create_reasoning_agent_from_provider(eval, provider_registry.get_solver(), mcp_registry)

    # Initialize conversation
    conversation = ChatHistory()
    initial_question = eval.question
    conversation.add_user_message(initial_question)

    mcps = []

    try:
        # Initialize solver agent (connects to MCP)
        await solver.initialize()

        asker_turns = eval.execution.asker.max_turns if eval.execution.asker else 1
        asker.chat_history.add_assistant_message(initial_question)  # Asker sees the initial question
        current_message = initial_question

        for asker_turn in range(asker_turns):
            # Solver responds (potentially using MCP tools)
            solver_response = await solver.message(current_message)
            conversation.add_assistant_message(solver_response.content or "")

            # Track MCP usage from connected clients
            for mcp_client in solver.mcp_clients:
                if await mcp_client.is_connected() and mcp_client.server_name not in mcps:
                    mcps.append(mcp_client.server_name)

            # If we've reached the limit of asker turns, stop
            if asker_turn >= asker_turns - 1:
                break

            try:
                next_question = await asker.message(solver_response.content or "")
            except Exception:
                # If asker generation fails, end the conversation
                break

            if not next_question or not isinstance(next_question, str):
                break

            conversation.add_user_message(next_question)
            current_message = next_question
    except Exception as e:
        return ConversationResult(conversation=conversation, mcps=mcps, success=False, error=str(e))
    finally:
        # Clean up MCP connections
        try:
            await solver.cleanup()
        except Exception as cleanup_error:
            # Log cleanup error but don't fail the conversation
            print(f"Warning: Error during cleanup: {cleanup_error}")

    return ConversationResult(conversation=conversation, mcps=mcps, success=True, solver_agent=solver)
