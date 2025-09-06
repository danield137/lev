import json
from unittest.mock import AsyncMock, Mock

import pytest

from lev.core.chat_history import ChatHistory
from lev.judge import EvaluationMode, Judge


class TestJudgeMatchMode:
    """Test traditional expected/actual scoring mode."""

    def setup_method(self):
        self.llm_provider = Mock()
        self.judge = Judge(self.llm_provider, EvaluationMode.MATCH)

    @pytest.mark.asyncio
    async def test_score_match_mode_basic(self):
        """Test basic match mode scoring."""
        conversation = ChatHistory()
        conversation.add_user_message("Tell me about Python programming")
        conversation.add_assistant_message(
            "Python is a programming language with key concepts like OOP and functional programming."
        )

        expected = {"topic": "python", "key_points": ["programming", "concepts"], "accuracy_indicators": ["python"]}

        result = await self.judge.score(expected=expected, conversation=conversation)

        assert result["mode"] == "match"
        assert "relevance" in result
        assert "completeness" in result
        assert "accuracy" in result
        assert "overall" in result


class TestJudgeExtractMode:
    """Test extract mode with scalar value extraction and comparison."""

    def setup_method(self):
        self.llm_provider = Mock()
        self.judge = Judge(self.llm_provider, EvaluationMode.EXTRACT)

    @pytest.mark.asyncio
    async def test_score_extract_mode_numeric_match(self):
        """Test extract mode with numeric value extraction and match."""
        conversation = ChatHistory()
        conversation.add_user_message("How many tables are in the database?")
        conversation.add_assistant_message("There are 104 tables in the database.")

        expected = 104

        # Mock LLM response for extraction
        mock_response = Mock()
        mock_response.content = "104"
        self.judge.llm_provider.chat_complete = AsyncMock(return_value=mock_response)

        result = await self.judge.score(conversation=conversation, expected=expected, mode=EvaluationMode.EXTRACT)

        assert result["mode"] == "extract"
        assert result["extracted"] == 104
        assert result["expected"] == 104
        assert result["match"] is True
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_score_extract_mode_numeric_tolerance(self):
        """Test extract mode with numeric tolerance."""
        conversation = ChatHistory()
        conversation.add_user_message("What is the temperature?")
        conversation.add_assistant_message("The temperature is 23.0001 degrees.")

        expected = 23.0

        # Mock LLM response for extraction
        mock_response = Mock()
        mock_response.content = "23.0001"
        self.judge.llm_provider.chat_complete = AsyncMock(return_value=mock_response)

        result = await self.judge.score(conversation=conversation, expected=expected, mode=EvaluationMode.EXTRACT)

        assert result["mode"] == "extract"
        assert result["extracted"] == 23.0001
        assert result["expected"] == 23.0
        assert result["match"] is True  # Within tolerance of 1e-3
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_score_extract_mode_string_match(self):
        """Test extract mode with string value extraction and match."""
        conversation = ChatHistory()
        conversation.add_user_message("What is the capital of France?")
        conversation.add_assistant_message("The capital of France is Paris.")

        expected = "Paris"

        # Mock LLM response for extraction
        mock_response = Mock()
        mock_response.content = "Paris"
        self.judge.llm_provider.chat_complete = AsyncMock(return_value=mock_response)

        result = await self.judge.score(conversation=conversation, expected=expected, mode=EvaluationMode.EXTRACT)

        assert result["mode"] == "extract"
        assert result["extracted"] == "Paris"
        assert result["expected"] == "Paris"
        assert result["match"] is True
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_score_extract_mode_string_case_insensitive(self):
        """Test extract mode with case-insensitive string matching."""
        conversation = ChatHistory()
        conversation.add_user_message("What is the status?")
        conversation.add_assistant_message("The status is ACTIVE.")

        expected = "active"

        # Mock LLM response for extraction
        mock_response = Mock()
        mock_response.content = "ACTIVE"
        self.judge.llm_provider.chat_complete = AsyncMock(return_value=mock_response)

        result = await self.judge.score(conversation=conversation, expected=expected, mode=EvaluationMode.EXTRACT)

        assert result["mode"] == "extract"
        assert result["extracted"] == "ACTIVE"
        assert result["expected"] == "active"
        assert result["match"] is True  # Case-insensitive match
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_score_extract_mode_no_match(self):
        """Test extract mode with no match."""
        conversation = ChatHistory()
        conversation.add_user_message("How many users are there?")
        conversation.add_assistant_message("There are 50 users in the system.")

        expected = 100

        # Mock LLM response for extraction
        mock_response = Mock()
        mock_response.content = "50"
        self.judge.llm_provider.chat_complete = AsyncMock(return_value=mock_response)

        result = await self.judge.score(conversation=conversation, expected=expected, mode=EvaluationMode.EXTRACT)

        assert result["mode"] == "extract"
        assert result["extracted"] == 50
        assert result["expected"] == 100
        assert result["match"] is False
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_score_extract_mode_missing_conversation(self):
        """Test that extract mode requires conversation."""
        with pytest.raises(ValueError, match="conversation and expected are required for EXTRACT"):
            await self.judge.score(expected=104, mode=EvaluationMode.EXTRACT)

    @pytest.mark.asyncio
    async def test_score_extract_mode_missing_expected(self):
        """Test that extract mode requires expected value."""
        conversation = ChatHistory()
        conversation.add_user_message("How many tables?")
        conversation.add_assistant_message("104 tables.")

        with pytest.raises(ValueError, match="conversation and expected are required for EXTRACT"):
            await self.judge.score(conversation=conversation, mode=EvaluationMode.EXTRACT)

    @pytest.mark.asyncio
    async def test_score_extract_mode_extraction_failure(self):
        """Test extract mode when LLM extraction fails."""
        conversation = ChatHistory()
        conversation.add_user_message("How many tables?")
        conversation.add_assistant_message("There are many tables.")

        expected = 104

        # Mock LLM to raise exception
        self.judge.llm_provider.chat_complete = Mock(side_effect=Exception("API Error"))

        result = await self.judge.score(conversation=conversation, expected=expected, mode=EvaluationMode.EXTRACT)

        assert result["mode"] == "extract"
        assert result["extracted"] is None
        assert result["expected"] == 104
        assert result["match"] is False
        assert result["score"] == 0.0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_score_match_mode_missing_params(self):
        """Test that match mode requires both conversation and expected."""
        conversation = ChatHistory()
        conversation.add_user_message("test")
        conversation.add_assistant_message("response")

        with pytest.raises(ValueError, match="conversation and expected are required for MATCH"):
            await self.judge.score(conversation=conversation, mode=EvaluationMode.MATCH)

        with pytest.raises(ValueError, match="conversation and expected are required for MATCH"):
            await self.judge.score(expected={"topic": "test"}, mode=EvaluationMode.MATCH)

    @pytest.mark.asyncio
    async def test_score_match_mode_explicit(self):
        """Test explicitly setting match mode."""
        conversation = ChatHistory()
        conversation.add_user_message("Tell me about testing")
        conversation.add_assistant_message("Testing is important for software quality.")

        expected = {"topic": "testing"}

        result = await self.judge.score(conversation=conversation, expected=expected, mode=EvaluationMode.MATCH)

        assert result["mode"] == "match"


class TestJudgeCritiqueMode:
    """Test critique mode with conversation analysis."""

    def setup_method(self):
        self.llm_provider = Mock()
        self.judge = Judge(self.llm_provider, EvaluationMode.CRITIQUE)

    @pytest.mark.asyncio
    async def test_score_critique_mode_basic(self):
        """Test basic critique mode evaluation."""
        conversation = ChatHistory()
        conversation.add_user_message("What is Python?")
        conversation.add_assistant_message("Python is a programming language...")

        # Mock LLM response for context compression
        mock_compress_response = Mock()
        mock_compress_response.content = "Python is a programming language..."

        # Mock LLM response for critique
        mock_critique_response = Mock()
        mock_critique_response.content = json.dumps(
            {"answered": True, "score": 0.9, "justification": "Good explanation of Python"}
        )

        self.judge.llm_provider.chat_complete = AsyncMock(return_value=mock_critique_response)
        self.judge.context_compressor.compress_chat = Mock(return_value="Python is a programming language...")

        result = await self.judge.score(conversation=conversation, mode=EvaluationMode.CRITIQUE)

        assert result["mode"] == "critique"
        assert result["answered"] is True
        assert result["score"] == 0.9
        assert "justification" in result

    @pytest.mark.asyncio
    async def test_score_critique_mode_missing_conversation(self):
        """Test that critique mode requires conversation."""
        with pytest.raises(ValueError, match="conversation is required for CRITIQUE"):
            await self.judge.score(mode=EvaluationMode.CRITIQUE)

    @pytest.mark.asyncio
    async def test_score_critique_mode_llm_failure(self):
        """Test critique mode when LLM call fails."""
        conversation = ChatHistory()
        conversation.add_user_message("What is Python?")
        conversation.add_assistant_message("Python is a programming language...")

        # Mock context compressor to return content
        self.judge.context_compressor.compress_chat = Mock(return_value="Python is a programming language...")

        # Mock LLM to raise exception
        self.judge.llm_provider.chat_complete = Mock(side_effect=Exception("API Error"))

        result = await self.judge.score(conversation=conversation, mode=EvaluationMode.CRITIQUE)

        assert result["mode"] == "critique"
        assert result["answered"] is False
        assert result["score"] == 0.0
        assert "Evaluation failed" in result["justification"]


class TestModeSelection:
    """Test evaluation mode selection and defaults."""

    def test_default_mode_match(self):
        """Test default mode is MATCH."""
        llm_provider = Mock()
        judge = Judge(llm_provider)
        assert judge.default_mode == EvaluationMode.MATCH

    def test_explicit_default_mode(self):
        """Test setting explicit default mode."""
        llm_provider = Mock()
        judge = Judge(llm_provider, EvaluationMode.CRITIQUE)
        assert judge.default_mode == EvaluationMode.CRITIQUE

    @pytest.mark.asyncio
    async def test_mode_override(self):
        """Test mode override works."""
        llm_provider = Mock()
        judge = Judge(llm_provider, EvaluationMode.MATCH)  # Default to MATCH

        conversation = ChatHistory()
        conversation.add_user_message("Test")
        conversation.add_assistant_message("Response")

        # Mock LLM response for critique mode
        mock_response = Mock()
        mock_response.content = json.dumps({"answered": True, "score": 0.8, "justification": "Good answer"})
        judge.llm_provider.chat_complete = AsyncMock(return_value=mock_response)
        judge.context_compressor.compress_chat = Mock(return_value="Response")

        # Override to CRITIQUE mode
        result = await judge.score(conversation=conversation, mode=EvaluationMode.CRITIQUE)
        assert result["mode"] == "critique"

    @pytest.mark.asyncio
    async def test_invalid_mode(self):
        """Test invalid mode raises error."""
        llm_provider = Mock()
        judge = Judge(llm_provider)

        conversation = ChatHistory()
        conversation.add_user_message("test")
        conversation.add_assistant_message("response")

        # Monkey patch to test invalid mode
        judge.default_mode = "invalid_mode"  # type: ignore

        with pytest.raises(ValueError, match="Unknown evaluation mode"):
            await judge.score(conversation=conversation, expected={"topic": "test"})


class TestMultipleEvaluationModes:
    """Test running multiple evaluation modes at once."""

    def setup_method(self):
        self.llm_provider = Mock()
        self.judge = Judge(self.llm_provider)

    @pytest.mark.asyncio
    async def test_multiple_modes_execution(self):
        """Test running multiple evaluation modes."""
        conversation = ChatHistory()
        conversation.add_user_message("What is Python?")
        conversation.add_assistant_message("Python is a programming language with object-oriented features.")

        expected = {"topic": "python", "key_points": ["programming", "language"]}

        # Mock for extract mode
        mock_extract_response = Mock()
        mock_extract_response.content = "Python"

        # Mock for critique mode
        mock_critique_response = Mock()
        mock_critique_response.content = json.dumps({"answered": True, "score": 0.85, "justification": "Good answer"})

        # Set up mocks
        self.judge.llm_provider.chat_complete = AsyncMock(side_effect=[mock_critique_response, mock_extract_response])
        self.judge.context_compressor.compress_chat = Mock(return_value="Python is a programming language...")

        # Run multiple modes
        modes = [EvaluationMode.MATCH, EvaluationMode.CRITIQUE, EvaluationMode.EXTRACT]
        result = await self.judge.score(conversation=conversation, expected=expected, mode=modes)

        # Should return results for all modes
        assert "match" in result
        assert "critique" in result
        assert "extract" in result

        assert result["match"]["mode"] == "match"
        assert result["critique"]["mode"] == "critique"
        assert result["extract"]["mode"] == "extract"


class TestBackwardCompatibility:
    """Test that the new API maintains some backward compatibility concepts."""

    @pytest.mark.asyncio
    async def test_conversation_based_api(self):
        """Test that the conversation-based API works."""
        llm_provider = Mock()
        judge = Judge(llm_provider)  # Default to MATCH mode

        conversation = ChatHistory()
        conversation.add_user_message("Tell me about Python programming")
        conversation.add_assistant_message("Python is a high-level programming language with dynamic typing.")

        expected = {"topic": "python", "key_points": ["programming"]}

        result = await judge.score(conversation=conversation, expected=expected)

        assert result["mode"] == "match"
        assert "relevance" in result
        assert "completeness" in result
        assert "accuracy" in result
        assert "overall" in result
