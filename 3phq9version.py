from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI
import random

# Load your OpenAI key from .env
load_dotenv()

# GPT-4o model
model = ChatOpenAI(model="gpt-4o")

# PHQ-9 Questions
phq9_questions = [
    "Over the last 2 weeks, how often have you had little interest or pleasure in doing things?",
    "Over the last 2 weeks, how often have you been feeling down, depressed, or hopeless?",
    "Over the last 2 weeks, how often have you had trouble falling or staying asleep, or sleeping too much?",
    "Over the last 2 weeks, how often have you felt tired or had little energy?",
    "Over the last 2 weeks, how often have you had poor appetite or been overeating?",
    "Over the last 2 weeks, how often have you felt bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Over the last 2 weeks, how often have you had trouble concentrating on things, such as reading or watching TV?",
    "Over the last 2 weeks, how often have you been moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you’ve been moving around a lot more than usual?",
    "Over the last 2 weeks, how often have you had thoughts that you would be better off dead or of hurting yourself in some way?"
]

# 🚨 Suicide/self-harm detection
trigger_phrases = [
    "suicidal", "kill myself", "want to die", "don't want to live",
    "hurt myself", "end my life", "took pills", "took sleeping pills"
]

# 🧠 Classification chain (fixed to include both question and response)
classification_prompt = ChatPromptTemplate.from_template(
    """
    You are a compassionate mental health assistant.
    Read the question: "{question}"
    Then read the user's response: "{response}"

    Based on both, classify the response as:
    "Not at all", "Several days", "More than half the days", or "Nearly every day".

    Output ONLY one of those phrases.
    """
)
classification_chain = classification_prompt | model | StrOutputParser()

# Empathetic response generation
empathetic_reply_chain = ChatPromptTemplate.from_template(
    """
    After reading:
    Q: {question}
    A: {response}
    Write one short, warm sentence validating what they shared.
    """
) | model | StrOutputParser()

# Final message generation
final_chain = ChatPromptTemplate.from_template(
    """
    The user shared:
    {all_responses}
    Their total PHQ-9 score is {score}.
    Write a short supportive message starting with:
    "Depression severity: ..."

    Then a gentle, kind paragraph (under 100 words).
    """
) | model | StrOutputParser()

# PHQ-9 scoring
score_mapping = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

# Confirmation phrases
confirmation_phrases = {
    "Not at all": [
        "I'll note that as 'not at all'.",
        "Thanks, marking 'not at all'.",
        "Got it — not at all."
    ],
    "Several days": [
        "Noted as 'several days'.",
        "I'll mark that down.",
        "Okay, several days it is."
    ],
    "More than half the days": [
        "I hear you — marking it accordingly.",
        "Noted: more than half the days.",
        "Understood, thank you."
    ],
    "Nearly every day": [
        "That sounds heavy — noted.",
        "Thanks for your honesty. Marking as 'nearly every day'.",
        "Got it, nearly every day."
    ]
}

# ✅ MAIN RUNNING CODE
print("\n🧠 Hi there — I'm here to listen and help.\nLet's go through 9 questions to check how you've been feeling.\nYou can respond in your own words.\n")

total_score = 0
user_answers = []
interrupted = False

for i, question in enumerate(phq9_questions):
    print(f"\n{i+1}. {question}")
    user_input = input("Your response: ").strip()

    # 🚨 Safety check
    if any(trigger in user_input.lower() for trigger in trigger_phrases):
        print("\n🚨 I'm really concerned about your safety.")
        print("You're not alone — and your feelings are valid.")
        print("Please talk to someone you trust, or a mental health professional.")
        print("If you're in Bangladesh, call 📞 13245 (Kaan Pete Roi — 24/7)")
        print("You matter. Help is available. 💛")
        interrupted = True
        break

    # ✨ Classify response based on question + answer
    classification = classification_chain.invoke({
        "question": question,
        "response": user_input
    }).strip()

    # 🎯 Score
    score = score_mapping.get(classification, 0)
    total_score += score
    user_answers.append((question, user_input, classification))

    # 💛 Confirmation + empathy
    confirm = random.choice(confirmation_phrases.get(classification, [f"Marked as {classification}."]))
    empathy = empathetic_reply_chain.invoke({"question": question, "response": user_input})

    print(f"\n👉 Interpreted as: {classification} ({score} point{'s' if score != 1 else ''})")
    print(f"{confirm}")
    print(f"{empathy}")

# 🟨 End summary
if interrupted:
    print("\nThe PHQ-9 test was stopped for your safety. Please take care. 💛")
else:
    summary = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in user_answers])
    final_message = final_chain.invoke({
        "all_responses": summary,
        "score": total_score
    })

    print("\n" + "="*60)
    print(final_message)
    print("="*60)
