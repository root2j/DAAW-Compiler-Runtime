import os
from groq import Groq
from dotenv import load_dotenv
from questionnaire import questionnaire

load_dotenv()

# ── Config

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

# ── System Prompt

PM_SYSTEM_PROMPT = """
You are an experienced Project Manager and Stakeholder who deeply understands software projects.
Your job is to take a user's project description, ask smart clarifying questions to fully 
understand their intent, and then produce a clear, structured project brief/draft.

Your behavior:
1. When given a project description, first ask 2-4 focused clarifying questions to fill any gaps.
   Keep questions concise and directly relevant — don't ask things already answered.
2. Once you have enough context (after the user answers your questions), produce a detailed draft.
   The draft should include:
   - Project Overview
   - Goals & Success Criteria
   - Key Features / Scope
   - Out of Scope (assumptions)
   - Open Questions / Risks
3. When refining, take the user's feedback seriously. Only rewrite the sections that need changing.
   Keep what was already approved implicitly. Be concise about what changed.

Always be professional but conversational. You're a smart collaborator, not a bureaucrat.
""".strip()

# ── Groq API Call

def call_agent(messages: list[dict]) -> str:
    """Send messages to the Groq model and return the assistant's reply."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()

# ── Clarification Phase

def clarification_phase(user_description: str) -> tuple[str, list[dict]]:
    """
    Agent asks clarifying questions, user answers them.
    Returns the final enriched context string and the message history up to this point.
    """
    print("\n" + "="*60)
    print("PM AGENT — Clarification Phase")
    print("="*60)

    # Seed the conversation with the user's description
    messages = [
        {"role": "system", "content": PM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Here is the project description from the user:\n\n{user_description}\n\n"
                "Please ask your clarifying questions now."
            )
        },
    ]

    # Agent asks questions
    agent_questions = call_agent(messages)
    print(f"\n[PM Agent]:\n{agent_questions}\n")
    messages.append({"role": "assistant", "content": agent_questions})

    # User answers
    print("[You]: ", end="")
    user_answers = input().strip()
    messages.append({"role": "user", "content": user_answers})

    return user_answers, messages

# ── Draft Generation 

def generate_draft(messages: list[dict]) -> str:
    """Ask the agent to produce the full project draft based on conversation so far."""
    messages.append({
        "role": "user",
        "content": (
            "Thank you for the answers. Now please produce the detailed project draft "
            "based on everything discussed."
        )
    })

    draft = call_agent(messages)
    return draft

# ── Refinement Loop

def refinement_loop(user_description: str, initial_draft: str) -> str:
    """
    Loops until user approves the draft.
    Only carries: system prompt + original description + latest draft + latest feedback.
    Returns the final approved draft.
    """
    print("\n" + "="*60)
    print("PM AGENT — Review & Refinement Phase")
    print("="*60)

    current_draft = initial_draft

    while True:
        print(f"\n[PM Agent — Draft]:\n{current_draft}\n")
        print("-"*60)
        print("Do you approve this draft?")
        print("  Type 'yes' to approve and move forward.")
        print("  Or describe what you'd like to change.")
        print("-"*60)
        print("[You]: ", end="")
        user_input = input().strip()

        if user_input.lower() in ("yes", "y"):
            print("\\ Draft approved! Moving to the next phase.\n")
            return current_draft

        # Refinement — minimal context window: system + original description + latest draft + feedback
        messages = [
            {"role": "system", "content": PM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Original project description:\n\n{user_description}"
            },
            {"role": "assistant", "content": current_draft},
            {
                "role": "user",
                "content": f"Please refine the draft based on this feedback:\n\n{user_input}"
            },
        ]

        print("\n[PM Agent is refining the draft...]\n")
        current_draft = call_agent(messages)

# Main Entrypoint

def run_pm_agent():

    user_description = questionnaire()

    if not user_description:
        print("No project description provided. Exiting.")
        return

    _, messages_after_clarification = clarification_phase(user_description)

    print("\n[PM Agent is generating the draft...]\n")
    initial_draft = generate_draft(messages_after_clarification)

    final_draft = refinement_loop(user_description, initial_draft)

    
    print("FINAL APPROVED DRAFT\n\n")
    print(final_draft)

    return final_draft


if __name__ == "__main__":
    run_pm_agent()