"""
Demo script to test the enhanced Judge functionality.
Shows both EXPECT and CRITIC modes, including RESET handling.
"""

import asyncio
import json
from unittest.mock import Mock

from lev.judge import EvaluationMode, Judge


def create_mock_llm_client():
    """Create a mock LLM client that returns realistic responses."""
    mock_client = Mock()

    def mock_create(**kwargs):
        # Extract the prompt to determine response
        messages = kwargs.get("messages", [])
        prompt = messages[0]["content"] if messages else ""

        # Simulate different responses based on the prompt content
        if "Python" in prompt and "programming language" in prompt:
            response_json = {
                "answered": True,
                "score": 0.9,
                "justification": "The assistant provided a comprehensive explanation of Python as a programming language.",
            }
        elif "Django" in prompt:
            response_json = {
                "answered": True,
                "score": 0.85,
                "justification": "Good explanation of Django web framework, though could include more details about its features.",
            }
        elif "incomplete" in prompt.lower():
            response_json = {
                "answered": False,
                "score": 0.3,
                "justification": "The response appears incomplete and doesn't fully address the user's question.",
            }
        else:
            response_json = {
                "answered": True,
                "score": 0.7,
                "justification": "The response adequately addresses the user's query.",
            }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(response_json)
        return mock_response

    mock_client.chat.completions.create = mock_create
    return mock_client


async def demo_expect_mode():
    """Demonstrate traditional EXPECT mode."""
    print("=== EXPECT MODE DEMO ===")

    llm_client = create_mock_llm_client()
    judge = Judge(llm_client, EvaluationMode.EXPECT)

    # Test case 1: Good match
    final_message = "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms."
    expected = {
        "topic": "python",
        "key_points": ["programming", "language", "simplicity"],
        "accuracy_indicators": ["high-level", "paradigms"],
    }

    result = await judge.score(final_message=final_message, expected=expected)
    print(f"Test 1 - Good match:")
    print(f"  Overall score: {result['overall']:.2f}")
    print(f"  Relevance: {result['relevance']:.2f}")
    print(f"  Completeness: {result['completeness']:.2f}")
    print(f"  Accuracy: {result['accuracy']:.2f}")
    print()

    # Test case 2: Poor match
    final_message = "The weather is nice today."
    result = await judge.score(final_message=final_message, expected=expected)
    print(f"Test 2 - Poor match:")
    print(f"  Overall score: {result['overall']:.2f}")
    print()


async def demo_critic_mode_basic():
    """Demonstrate basic CRITIC mode."""
    print("=== CRITIC MODE DEMO (Basic) ===")

    llm_client = create_mock_llm_client()
    judge = Judge(llm_client, EvaluationMode.CRITIC)

    # Test case 1: Good conversation
    conversation = [
        {"role": "user", "content": "What is Python and why is it popular?"},
        {
            "role": "assistant",
            "content": "Python is a high-level programming language that's popular because of its simple syntax, extensive libraries, and versatility in applications like web development, data science, and AI.",
        },
    ]

    result = await judge.score(conversation=conversation)
    print(f"Test 1 - Good conversation:")
    print(f"  Answered: {result['answered']}")
    print(f"  Score: {result['score']}")
    print(f"  Justification: {result['justification']}")
    print()

    # Test case 2: No assistant response
    conversation = [{"role": "user", "content": "What is Python?"}]

    result = await judge.score(conversation=conversation)
    print(f"Test 2 - No assistant response:")
    print(f"  Answered: {result['answered']}")
    print(f"  Score: {result['score']}")
    print(f"  Justification: {result['justification']}")
    print()


async def demo_reset_handling():
    """Demonstrate RESET scenario handling."""
    print("=== CRITIC MODE DEMO (RESET Handling) ===")

    llm_client = create_mock_llm_client()
    judge = Judge(llm_client, EvaluationMode.CRITIC)

    # Test case 1: Conversation with RESET
    conversation = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."},
        {"role": "user", "content": "Can you tell me more about its history?"},
        {"role": "assistant", "content": "Python was created by Guido van Rossum..."},
        {"role": "user", "content": "RESET"},
        {"role": "user", "content": "Tell me about Django"},
        {
            "role": "assistant",
            "content": "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design.",
        },
    ]

    # Test the query synthesis
    synthesized_query = judge._synthesize_final_query(conversation)
    print(f"Test 1 - RESET scenario:")
    print(f"  Synthesized query: {synthesized_query}")

    result = await judge.score(conversation=conversation)
    print(f"  Answered: {result['answered']}")
    print(f"  Score: {result['score']}")
    print(f"  Justification: {result['justification']}")
    print()

    # Test case 2: Multiple clarifications without RESET
    conversation = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Can you give examples?"},
        {"role": "user", "content": "Specifically for image recognition?"},
        {
            "role": "assistant",
            "content": "Sure! For image recognition, common ML approaches include convolutional neural networks (CNNs), transfer learning with pre-trained models, and computer vision techniques.",
        },
    ]

    synthesized_query = judge._synthesize_final_query(conversation)
    print(f"Test 2 - Multiple clarifications:")
    print(f"  Synthesized query: {synthesized_query}")

    result = await judge.score(conversation=conversation)
    print(f"  Answered: {result['answered']}")
    print(f"  Score: {result['score']}")
    print()


async def demo_backward_compatibility():
    """Demonstrate backward compatibility."""
    print("=== BACKWARD COMPATIBILITY DEMO ===")

    llm_client = create_mock_llm_client()
    judge = Judge(llm_client)  # Default mode (EXPECT)

    # Old-style API call (positional arguments)
    final_message = "Python is a programming language used for web development and data science."
    expected = {"topic": "python", "key_points": ["programming", "web", "data"]}

    result = await judge.score(final_message, expected)
    print(f"Old-style API call:")
    print(f"  Mode: {result['mode']}")
    print(f"  Overall score: {result['overall']:.2f}")
    print()


async def main():
    """Run all demos."""
    print("Judge Enhancement Demo")
    print("=" * 50)
    print()

    await demo_expect_mode()
    await demo_critic_mode_basic()
    await demo_reset_handling()
    await demo_backward_compatibility()

    print("Demo completed! All functionality working as expected.")


if __name__ == "__main__":
    asyncio.run(main())
