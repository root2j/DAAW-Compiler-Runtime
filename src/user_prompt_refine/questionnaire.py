QUESTIONS = [
    {
        "id": "task",
        "label": "What's the task or activity you want to automate?",
        "hint": "e.g. 'when someone fills a form, send them a welcome email and add them to my CRM'",
    },
    {
        "id": "trigger",
        "label": "What triggers this workflow?",
        "hint": "e.g. a form submission, a scheduled time, an incoming email, a webhook, manually",
    },
    {
        "id": "apps",
        "label": "What apps or services are involved?",
        "hint": "e.g. Gmail, Slack, Google Sheets, Notion, Typeform — list all you can think of",
    },
    {
        "id": "outcome",
        "label": "What should happen at the end of the workflow? What does 'done' look like?",
        "hint": "e.g. a notification is sent, a row is added, a file is created",
    },
    {
        "id": "frequency",
        "label": "How often do you expect this workflow to run?",
        "hint": "e.g. once a day, every time an event happens, hundreds of times a day",
    },
    {
        "id": "conditions",
        "label": "Are there any conditions or special cases the workflow should handle?",
        "hint": "e.g. 'only send the email if the country is India', 'skip if the row already exists' — type 'none' if not applicable",
    },
    {
        "id": "manual_process",
        "label": "Have you tried automating this before, or are you currently doing this manually? If manually, walk me through the steps you take.",
        "hint": "Be as detailed as possible — this helps us understand edge cases you might not think to mention",
    },
]


def questionnaire() -> str:
    """
    Walks the user through 7 questions one at a time.
    Returns a formatted string summary of all answers for the PM agent.
    """
    print("\n" + "="*60)
    print("  WORKFLOW AUTOMATION — Project Intake")
    print("="*60)
    print("Answer the following questions to help us understand")
    print("what you want to automate. Take your time.\n")

    answers = {}

    for i, q in enumerate(QUESTIONS, start=1):
        print(f"Question {i} of {len(QUESTIONS)}")
        print(f"  {q['label']}")
        print(f"  Hint: {q['hint']}")
        print()

        while True:
            print("  Your answer: ", end="")
            answer = input().strip()
            if answer:
                break
            print("  (Answer can't be empty, please type something)\n")

        answers[q["id"]] = answer
        print()

    # Format answers into a clean summary string for the PM agent
    summary_lines = ["User's project intake answers:\n"]
    for i, q in enumerate(QUESTIONS, start=1):
        summary_lines.append(f"Q{i}. {q['label']}")
        summary_lines.append(f"A{i}. {answers[q['id']]}")
        summary_lines.append("")

    return "\n".join(summary_lines)


if __name__ == "__main__":
    # Quick test run
    result = questionnaire()
    print("="*60)
    print("COLLECTED DESCRIPTION:")
    print("="*60)
    print(result)