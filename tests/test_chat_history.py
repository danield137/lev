from lev.core.chat_history import ChatHistory


def test_chat_history_initialization():
    """Test ChatHistory initializes correctly."""
    history = ChatHistory()
    assert history.messages == []
    assert history.tool_calls == []


def test_add_user_message():
    """Test adding user messages."""
    history = ChatHistory()
    history.add_user_message("Hello")

    assert len(history.messages) == 1
    assert history.messages[0]["role"] == "user"
    assert history.messages[0]["content"] == "Hello"


def test_add_assistant_message():
    """Test adding assistant messages."""
    history = ChatHistory()
    history.add_assistant_message("Hi there!")

    assert len(history.messages) == 1
    assert history.messages[0]["role"] == "assistant"
    assert history.messages[0]["content"] == "Hi there!"


def test_add_tool_call():
    """Test adding tool calls."""
    history = ChatHistory()
    arguments = {"param": "value"}
    result = {"success": True}
    server_name = "test-server"

    history.add_tool_call(server_name, "test_tool", arguments, result)

    assert len(history.tool_calls) == 1
    tool_call = history.tool_calls[0]
    assert tool_call["tool_name"] == "test_tool"
    assert tool_call["arguments"] == arguments
    assert tool_call["result"] == result
    assert tool_call["server_name"] == server_name
    assert "timestamp" in tool_call


def test_get_conversation():
    """Test getting the conversation history."""
    history = ChatHistory()
    history.add_user_message("Hello")
    history.add_tool_call("server1", "tool1", {}, {"message": "Success"})
    history.add_tool_response_message("tool_call_id_1", "Tool response")
    history.add_assistant_message("Hi there!")

    conversation = history.get_conversation()
    assert len(conversation) == 2
    assert conversation[0].role == "user"
    assert conversation[0].content == "Hello"
    assert conversation[1].role == "assistant"
    assert conversation[1].content == "Hi there!"
