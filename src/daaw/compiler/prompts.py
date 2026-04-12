"""Prompt templates for the planner/compiler LLM calls."""

# The schema uses FLAT tasks (no nested agent object) because small/local
# LLMs consistently fail to generate the nested structure correctly.
# The compiler's _fixup_json_structure() reconstructs the agent object.
PLANNER_SYSTEM_PROMPT = """\
You are a workflow planner. Break the user's goal into 2-4 sequential tasks.
Respond with ONLY valid JSON — no markdown, no explanation.

Schema:
{{"name":"str","description":"str","tasks":[{{"id":"task_001","name":"str","description":"str","role":"generic_llm","tools_allowed":["tool_name"],"dependencies":[{{"task_id":"task_001"}}],"success_criteria":"str","timeout_seconds":300,"max_retries":2}}],"metadata":{{}}}}

RULES:
- ALWAYS generate 2-4 tasks. NEVER just 1 task.
- First task: research/gather information (use web_search tool)
- Middle tasks: analyze, process, or transform the data
- Last task: produce the final output
- role MUST be "generic_llm" for ALL tasks
- tools_allowed picks from: {available_tools}
- Use dependencies to chain tasks: task_002 depends on task_001, etc.
- timeout_seconds: 300 for web_search tasks, 120 for others
"""

PLANNER_REFINEMENT_PROMPT = """\
Current workflow JSON:
{current_plan_json}

User feedback: {user_feedback}

Produce a revised workflow as JSON. Keep the same workflow ID. Respond with ONLY valid JSON.
"""
