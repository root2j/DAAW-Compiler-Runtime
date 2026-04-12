"""Prompt templates for the Critic LLM calls."""

CRITIC_SYSTEM_PROMPT = """\
You evaluate task outputs. Respond with ONLY valid JSON:
{{"verdict":"pass" or "fail","reasoning":"brief explanation","patch":null}}

Be LENIENT: if the task produced relevant content addressing the goal, verdict is "pass".
Only "fail" if the output is empty, completely off-topic, or an error message.
"""

CRITIC_EVALUATION_PROMPT = """\
Task: {task_name}
Goal: {task_description}
Criteria: {success_criteria}
Status: {status} | Time: {elapsed:.1f}s

Output (first 800 chars):
{task_output}

JSON verdict:"""
