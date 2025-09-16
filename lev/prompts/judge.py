"""
Judge subsystem prompts for evaluation and scoring.
"""

# TODO: consider other promts
# https://github.com/Accenture/mcp-bench/blob/main/benchmark/evaluator.py
# https://huggingface.co/learn/cookbook/en/llm_judge
JUDGE_CRITIQUE_USER_PROMPT_TEMPLATE = """
You are evaluating whether an assistant's response adequately answered a user's query.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Important evaluation rules:
- If the system_answer provides a valid query, detailed reasoning, and internally consistent results, assume it is correct unless you can point to a clear logical or arithmetic mistake.
- Do not downgrade answers just because the reported values seem unusually large or small; rely only on consistency and correctness of reasoning shown.
- If tool calls are shown in the conversation (denoted with [tool:{{servername}}.{{toolname}}]({{args_list}}).>), assume they are real and their responses truthful.
- Your task is to judge adequacy and helpfulness relative to the user query, not to re-run the dataset or fact-check external sources.

If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.

If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.

reply with the following JSON ONLY:
{{
  "answered": true/false,
  "score": 0.0-1.0,
  "justification": "concise and to the point reason for the score."
}}

---
USER QUERY:
{user_query}
===
CONVERSATION:
{conversation}
===
TOOL CALLS:
{tool_calls_trace}
"""


JUDGE_EXTRACT_USER_PROMPT_TEMPLATE = """Extract ONLY the scalar value that answers the question. Return just the value.

Question: {question}

Answer: {answer}"""
