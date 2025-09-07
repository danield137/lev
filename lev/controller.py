import json
from lev.host.mcp_host import McpHost
from lev.common.roles import MessageRole
from lev.core.agent import Agent
from lev.prompts.reasoning import REASONING_AGENT_ANSWER_VALIDATION_PROMPT


class Introspector:
    def __init__(self, agent: Agent):
        self.agent = agent

    async def validate(self, conversation_history: str, response_to_validate: str) -> dict:
        """
        Validate if an answer is complete and correct.
        Returns dict with 'valid' bool and optional 'followup' message.
        """
        if not self.agent:
            return {"valid": True}

        try:
            prompt = REASONING_AGENT_ANSWER_VALIDATION_PROMPT.format(
                conversation_history=conversation_history,
                response_to_validate=response_to_validate,
            )

            await self.agent.reset()
            introspect_resp = await self.agent.message(prompt)

            try:
                decision = json.loads(introspect_resp.content or "{}")
                if not decision.get("valid", True):
                    followup = decision.get("followup_question", "Please provide more details.")
                    return {"valid": False, "followup": followup}
                else:
                    return {"valid": True}
            except json.JSONDecodeError:
                return {"valid": True}  # Default to valid if JSON parse fails

        except Exception as e:
            return {"valid": True}  # Default to valid on error

    async def plan_next(self, conversation_history: str) -> dict:
        """
        Plan the next step after tools have been executed.
        Returns dict with 'continue' bool and optional 'next_prompt'.
        """
        if not self.agent:
            return {"continue": False}

        try:
            await self.agent.reset()
            introspect_resp = await self.agent.message(conversation_history)

            try:
                decision = json.loads(introspect_resp.content or "{}")
                should_continue = decision.get("continue", False)
                next_prompt = decision.get("next_prompt")
                reason = decision.get("reason", "")
                return {"continue": should_continue, "next_prompt": next_prompt, "reason": reason}
            except json.JSONDecodeError:
                return {"continue": False}

        except Exception as e:
            return {"continue": False}


class Controller:
    host: McpHost
    introspector: Introspector

    def __init__(self, host: McpHost, introspector: Introspector, *, max_steps: int = 8):
        self.host, self.introspector, self.max_steps = host, introspector, max_steps

    async def run(self, question: str) -> str:
        done = False
        await self.host.reset()
        role, prompt = MessageRole.USER, question
        for _ in range(self.max_steps):
            turn = await self.host.step(prompt, role=role)
            if turn.fatal_error:
                return f"HostError: {turn.fatal_error}"

            if not turn.had_tools: # only follow up if previous introspection did not declare completion
                if done:
                    return turn.content or ""
                v = await self.introspector.validate(self.host.history().render_trace(), turn.content or "")
                if v.get("valid", True):
                    return turn.content or ""
                role, prompt = MessageRole.DEVELOPER, v.get("followup", "Clarify and answer precisely.")
                continue

            # Tools were executed by Host. Decide next message.
            if self.introspector:
                plan = await self.introspector.plan_next(self.host.history().render_trace())
                if plan.get("continue", False):
                    role, prompt = MessageRole.DEVELOPER, plan.get("next_prompt", "Proceed.")
                    continue

            # Default synthesis instruction when no explicit plan.
            role, prompt = MessageRole.DEVELOPER, "Synthesize the final answer using the tool results."
            done = True

        # Get the last assistant message content
        history = self.host.history()
        if history.messages:
            for msg in reversed(history.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return msg["content"]
        return "No final answer."
