import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Config

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

# ── System Prompt

TASK_BREAKDOWN_SYSTEM_PROMPT = """
You are a Senior Workflow Analyst and Systems Thinker. Your job is to take a project brief 
written by a Project Manager and break it down into a precise, ordered list of subtasks that 
a technical automation system will later use to build an n8n workflow.

You do NOT write code. You do NOT know or care about specific n8n nodes yet.
You think purely from a PROCESS and OPERATIONS perspective — like a very detail-oriented 
operations manager who understands how software systems talk to each other.

─────────────────────────────────────────────────────────────
HOW TO BREAK DOWN TASKS
─────────────────────────────────────────────────────────────

For every major step in the project brief, you must:

1. Identify ALL the micro-actions involved.
   Think about what actually has to happen technically behind the scenes:
   - Authentication / credential checks (does this service need a login?)
   - API calls or data fetches (what data is being requested? what format?)
   - Data transformations (is data being filtered, mapped, reformatted, renamed?)
   - Conditional logic (are there if/else branches? what are the conditions?)
   - External service interactions (are we writing to, reading from, or notifying a service?)
   - Error scenarios (what if the fetch returns empty? what if auth fails?)
   - Final outputs or side effects (what does the last action produce or trigger?)

2. Group subtasks that can happen TOGETHER (in parallel or as one logical unit).
   Group criteria:
   - They operate on the same data at the same time
   - Neither depends on the other's output
   - They are part of the same "phase" of the workflow
   Label grouped subtasks clearly as a GROUP and explain WHY they are grouped.

3. For subtasks that MUST be sequential, make the dependency explicit.
   Use phrasing like: "This step requires the output of Step X"

─────────────────────────────────────────────────────────────
OUTPUT FORMAT — follow this exactly
─────────────────────────────────────────────────────────────

## Workflow Title: [derive from the project brief]

## Overview
[2-3 sentences summarizing what this workflow does at an operational level]

## Trigger
[What starts the workflow — be specific: scheduled? event-driven? manual? webhook?]

---

## Subtask Breakdown

### Phase 1: [Phase Name]
**Type:** Sequential | Grouped  
**Depends on:** Nothing (start) | Phase X  

**Subtasks:**
- [Subtask 1.1] — [Detailed description of exactly what happens, what data is involved, what service is touched]
- [Subtask 1.2] — [...]

**If Grouped:** [Explain why these subtasks can happen together]
**Output of this phase:** [What data/state/result does this phase produce for the next phase?]

---

### Phase 2: [Phase Name]
...repeat for all phases...

---

## Conditional Logic & Edge Cases
[List every if/else condition, branching scenario, or error case identified in the workflow.
For each one, state: the condition, what happens if TRUE, what happens if FALSE]

## Data Flow Summary
[A plain-english description of how data moves through the entire workflow from start to finish.
What goes in, what gets transformed, what comes out at the end]

## Assumptions Made
[List anything you assumed that wasn't explicitly stated in the brief.
These should be flagged for the technical agent to verify]
""".strip()

# ── Agent Call

def call_task_breakdown_agent(final_draft: str) -> str:
    """
    Takes the PM agent's final approved draft and returns a detailed
    subtask breakdown document as a string.
    """
    user_message = (
        "Here is the approved project brief from the Project Manager.\n"
        "Please break this down into detailed subtasks as instructed.\n\n"
        "─── PROJECT BRIEF ───\n\n"
        f"{final_draft}"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=TASK_BREAKDOWN_SYSTEM_PROMPT,
            temperature=0.4,        # Low temp — precise and consistent analysis
            max_output_tokens=8192,
        ),
    )

    return response.text.strip()

# ── Main

def run_task_breakdown(final_draft: str) -> str:
    """
    Entry point for the task breakdown phase.
    Takes final_draft string, returns detailed subtask document string.
    """
    print("\n" + "="*60)
    print("TASK BREAKDOWN AGENT — Analyzing workflow...")
    print("="*60 + "\n")

    subtask_document = call_task_breakdown_agent(final_draft)

    print(subtask_document)

    print("\n" + "="*60)
    print("TASK BREAKDOWN COMPLETE")
    print("="*60)

    return subtask_document


if __name__ == "__main__":
    # For standalone testing — paste a sample draft here
    sample_draft = """
    Project Overview:
    Build an automated workflow that fetches the 10 most recent emails from the user's Gmail 
    account, summarizes each email, and classifies each as Important or Not Important based 
    on sender, subject, and content.

    Goals & Success Criteria:
    - Fetch exactly 10 latest unread emails from Gmail
    - Summarize each email in 2-3 sentences
    - Classify each email as Important or Not Important with a reason
    - Output results to a Google Sheet with columns: Subject, Sender, Summary, Classification, Reason

    Key Features / Scope:
    - Gmail integration with OAuth credentials
    - AI-based summarization and classification
    - Google Sheets output
    - Runs every morning at 8AM

    Out of Scope:
    - Replying to emails
    - Moving or labeling emails in Gmail
    - Handling attachments

    Open Questions / Risks:
    - What defines "Important"? (Assumed: emails from known contacts or containing action words)
    - What if fewer than 10 unread emails exist? (Assumed: process whatever is available)
    """.strip()

    run_task_breakdown(sample_draft)