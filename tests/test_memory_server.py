#!/usr/bin/env python3

import json
import tempfile
from pathlib import Path

from fw_context_server.memory_server import MemoryServer


def test_memory_server():
    """Test the MemoryServer functionality."""
    # Use a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        server = MemoryServer(data_dir=temp_dir)

        # Test storing memories
        print("Testing store_memory...")
        result1 = server.store_memory(
            content="This is my first memory about Python programming",
            time="2025-01-19T21:30:00Z",
            tags=["python", "programming", "learning"],
        )
        print(f"Store result 1: {result1}")

        result2 = server.store_memory(
            content="Remember to buy groceries tomorrow", time="2025-01-19T21:31:00Z", tags=["todo", "shopping"]
        )
        print(f"Store result 2: {result2}")

        result3 = server.store_memory(
            content="Meeting with client about Python project",
            time="2025-01-19T21:32:00Z",
            tags=["meeting", "client", "python"],
        )
        print(f"Store result 3: {result3}")

        # Test recalling memories
        print("\nTesting recall_memory...")

        # Search for "python"
        python_memories = server.recall_memory("python")
        print(f"Memories containing 'python': {python_memories}")

        # Search for "shopping" (should find it in tags)
        shopping_memories = server.recall_memory("shopping")
        print(f"Memories containing 'shopping': {shopping_memories}")

        # Search for something that doesn't exist
        nonexistent_memories = server.recall_memory("nonexistent")
        print(f"Memories containing 'nonexistent': {nonexistent_memories}")

        # Verify files were created
        data_path = Path(temp_dir)
        json_files = list(data_path.glob("*.json"))
        print(f"\nCreated {len(json_files)} JSON files")

        # Show content of one file
        if json_files:
            with open(json_files[0]) as f:
                sample_content = json.load(f)
            print(f"Sample file content: {sample_content}")


if __name__ == "__main__":
    test_memory_server()
