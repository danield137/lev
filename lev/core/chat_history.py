import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast


@dataclass(slots=True)
class ParticipantMessage:
    role: str
    content: str
    timestamp: str


class ChatHistory:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index) -> dict[str, Any]:
        return self.messages[index]

    def __iter__(self):
        return iter(self.messages)

    def add_system_message(self, content: str):
        """Add a system message to history."""
        self.messages.append({"role": "system", "content": content, "timestamp": datetime.now().isoformat()})

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append({"role": "user", "content": content, "timestamp": datetime.now().isoformat()})

    def add_assistant_message(self, content: str):
        """Add an assistant message to history."""
        self.messages.append({"role": "assistant", "content": content, "timestamp": datetime.now().isoformat()})

    def add_assistant_tool_call_message(self, content: str, tool_calls: list[Any]):
        """Add an assistant message with tool calls to history."""
        self.messages.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "timestamp": datetime.now().isoformat()}
        )

    def add_tool_response_message(self, tool_call_id: str, content: str):
        """Add a tool response message to history."""
        self.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content, "timestamp": datetime.now().isoformat()}
        )

    def add_tool_call(self, server_name: str, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]):
        """Record a tool call for context."""
        self.tool_calls.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "server_name": server_name,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_conversation(
        self,
        with_participants: bool | None = True,
        with_system: bool | None = False,
        with_tools: bool | None = False,
    ) -> list[ParticipantMessage]:
        if not with_participants and not with_system and not with_tools:
            raise ValueError("At least one message type must be included.")
        roles = []
        if with_participants:
            roles += ["user", "assistant"]
        if with_system:
            roles += ["system", "developer", "platform"]
        if with_tools:
            roles += ["tool", "tool_calls"]
        return [
            ParticipantMessage(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
            for msg in self.messages
            if msg.get("role") in roles
        ]

    def get_user_messages(self) -> list[ParticipantMessage]:
        """Get all user messages from the conversation."""
        return [ParticipantMessage(**msg) for msg in self.messages if msg.get("role") == "user"]

    def get_assistant_messages(self) -> list[ParticipantMessage]:
        """Get all assistant messages from the conversation."""
        return [ParticipantMessage(**msg) for msg in self.messages if msg.get("role") == "assistant"]

    def to_role_content_messages(self, with_system: bool = False, with_tools: bool = False) -> list[dict[str, Any]]:
        messages = []
        for msg in self.messages:
            if msg["role"] == "user":
                messages.append({"role": msg["role"], "content": msg["content"]})
            elif msg["role"] == "assistant":
                # Handle assistant messages with or without tool calls
                message = {"role": msg["role"], "content": msg["content"]}
                if with_tools and "tool_calls" in msg:
                    message["tool_calls"] = msg["tool_calls"]
                messages.append(message)
            elif msg["role"] == "tool" and with_tools:
                # Tool response messages need tool_call_id
                messages.append({"role": msg["role"], "content": msg["content"], "tool_call_id": msg["tool_call_id"]})
            elif msg["role"] in ["system", "developer", "platform"] and with_system:
                messages.append({"role": msg["role"], "content": msg["content"]})
        return messages

    def format_message_for_console(self, content: str, max_length: int = 120) -> str:
        """Format a message for console output with length trimming."""
        if len(content) <= max_length:
            return content
        trimmed = content[: max_length - 3]
        return trimmed + f"... ({len(content.split())-len(trimmed.split())} tokens excluded)"

    def format_tool_call_for_console(self, tool_call: dict[str, Any], max_length: int = 120) -> str:
        """Format a tool call for console output."""
        tool_name = tool_call.get("tool_name", "unknown")
        server_name = tool_call.get("server_name", "unknown")
        arguments = tool_call.get("arguments", {})
        result = tool_call.get("result", {})

        # Format arguments more readably
        if arguments:
            args_str = str(arguments)
            if len(args_str) > 60:
                # Truncate long argument values
                truncated_args = {}
                for k, v in arguments.items():
                    if isinstance(v, str) and len(v) > 40:
                        truncated_args[k] = v[:40] + "..."
                    else:
                        truncated_args[k] = v
                args_str = str(truncated_args)
        else:
            args_str = "{}"

        # Show result summary if available
        result_summary = ""
        if result and isinstance(result, dict):
            if "success" in result:
                result_summary = f" → {'✓' if result.get('success') else '✗'}"
            elif "text" in result:
                text = result["text"][:30] + "..." if len(result.get("text", "")) > 30 else result.get("text", "")
                result_summary = f" → '{text}'"

        full_str = f"{server_name}.{tool_name}({args_str}){result_summary}"

        # Final length check
        if len(full_str) > max_length:
            trimmed = full_str[: max_length - 3]
            full_str = trimmed + f"... ({len(full_str.split())-len(trimmed.split())} tokens excluded)"

        return full_str

    def _to_str(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return str(content)

    def render_trace(
        self,
        max_preview_len: int = 100,
    ) -> str:
        """Render the entire conversation as a console-style trace with back-and-forth turns.

        Rules:
        - Print every user message as:  USER      → <content>
        - For assistant tool use:
            ASSISTANT → <ns.func(args)>
                      ← <tool_response_preview>
          Then the assistant's spoken message as:
                      💬 <assistant message>
        - If the assistant replies without tools:
            ASSISTANT 💬 <assistant message>
        - Parameters first_user_message/final_assistant_message are kept for backward compat but not required.
        """
        lines: list[str] = []
        cont = "          "  # fixed 10-space continuation indent
        assistant_prefix_active = False  # within an assistant block (tool calls/responses)

        for _msg in self.messages:
            msg = cast(dict[str, Any], _msg)
            role = msg.get("role", "")

            if role == "user":
                content = self._to_str(msg.get("content", "") or "")
                lines.append(f"USER      → {content}")
                assistant_prefix_active = False

            elif role == "assistant":
                # Tool call phase
                tool_calls = msg.get("tool_calls") or []
                if isinstance(tool_calls, list) and tool_calls:
                    for tc in tool_calls:
                        func = (tc.get("function", {}) or {}) if isinstance(tc, dict) else {}
                        name = func.get("name", "unknown")
                        # Extract server name from stored tool calls if available
                        server_name = "unknown"
                        tool_call_id = tc.get("id", "")
                        for stored_call in self.tool_calls:
                            if stored_call.get("tool_name") == name:
                                server_name = stored_call.get("server_name", "unknown")
                                break
                        full_name = f"[tool_call:{server_name}.{name}]"

                        raw_args = func.get("arguments", "") or ""
                        try:
                            if isinstance(raw_args, str):
                                parsed_args = json.loads(raw_args)
                            else:
                                parsed_args = raw_args
                            if isinstance(parsed_args, dict):
                                args_str = ", ".join([f'{k}="{v}"' for k, v in parsed_args.items()])
                            else:
                                args_str = self._to_str(parsed_args)
                        except Exception:
                            args_str = self._to_str(raw_args)

                        prefix = "ASSISTANT → " if not assistant_prefix_active else cont
                        lines.append(f"{prefix}{full_name}({args_str})")
                        assistant_prefix_active = True

                # Assistant spoken content (if any)
                content_any = msg.get("content", "") or ""
                if content_any:
                    text = self._to_str(content_any)
                    if assistant_prefix_active:
                        lines.append(f"{cont}💬 {text}")
                    else:
                        lines.append(f"ASSISTANT 💬 {text}")
                    assistant_prefix_active = False

            elif role == "tool":
                # Tool response preview under assistant block
                preview = self._to_str(msg.get("content", "") or "")
                if len(preview) > max_preview_len:
                    trimmed = preview[:max_preview_len]
                    preview = trimmed + f"... ({len(preview.split())-len(trimmed.split())} tokens excluded)"
                lines.append(f"{cont}← {preview}")
                # remain in assistant block

        return "\n".join(lines)
