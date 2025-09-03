"""
Context compressor subsystem prompts for conversation compression.
"""

CONTEXT_COMPRESSOR_COMPRESS_USER_PROMPT_TEMPLATE = """
You are a smart context compressor. Your job is to take a conversation, and compress it to the most concise form possible. 
Follow these rules:
* Aim for < 200 words. Focus on preserving the user's intent and key details. 
* If you encounter long responses like files, table, etc... just include a placeholder with a one line summary.
* For sections that contain technical information (traces, error, etc..) - keep just critical information.
* IT IS CRITICAL THAT EACH MESSAGE RETAINS ITS ORIGINAL ROLE. Keep existing structure (e.g. USER: .., ASSISTANT: ..). 
Below is the conversation history, respond with the compressed context only.

'''
{message_sequence}
'''
"""

CONTEXT_COMPRESSOR_COMPRESS_USER_PROMPT2_TEMPLATE = """
You are a smart context compressor. Your job is to take a PROMPT, and compress it to the most concise form possible, that still retains its original meaning.
Follow these rules:
* Aim for < 200 words. Focus on preserving the intent and key details.
* IT IS CRITICAL THAT EACH RETAIN ROLES AS THEY APPEAR IN THE ORIGINAL TEXT. Keep existing structure (e.g. USER: .., ASSISTANT: ..). 
* If you encounter long sections like files, table, etc... just include a placeholder with a one line summary.
* For sections that contain technical information (traces, error, etc..) - keep just critical information.
Below is the conversation history, respond with the compressed context only.

'''
{message_sequence}
'''
"""
