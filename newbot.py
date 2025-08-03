from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI
import random

# Load API key from .env
load_dotenv()

# Initialize model
model = ChatOpenAI(model="gpt-4o")

# PHQ-9 questions
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

# Classification prompt
classification_prompt = ChatPromptTemplate.from_template(
    """
    You are a compassionate mental health assistant helping assess depression symptoms.
    Based on the user's detailed response: "{response}", classify their experience as one of the following:
    "Not at all", "Several days", "More than half the days", or "Nearly every day".
    Only output the matching phrase exactly.
    """
)

classification_chain = classification_prompt | model | StrOutputParser()

# Empathetic reply prompt
empathetic_reply_prompt = ChatPromptTemplate.from_template(
    """
    You are a kind and gentle professional. After reading this question:
    "{question}"
    And hearing this answer:
    "{response}"
    Write a short, empathetic one-sentence reply that validates and gently reflects on what they shared. Be brief and caring.
    """
)

empathetic_reply_chain = empathetic_reply_prompt | model | StrOutputParser()

# Final message summary prompt
final_message_prompt = ChatPromptTemplate.from_template(
    """
    The user shared these feelings:
    {all_responses}

    Their total PHQ-9 score is {score}.
    Based on this score and their descriptions, generate a short, warm, emotionally supportive paragraph.
    Start by stating the depression severity clearly (e.g., "Depression severity: Moderately severe").
    Then write a kind message that reflects understanding and reassurance based on the user's emotional experience.
    Important: Keep it under 100 words.
    """
)

final_chain = final_message_prompt | model | StrOutputParser()

# Score mapping
score_mapping = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

# Confirmation phrases for conversational flow
confirmation_phrases = {
    "Not at all": [
        "I understand, I'll note that as 'not at all'.",
        "Thanks for sharing, I'll put that down as 'not at all'.",
        "Got it — I'll classify that as 'not at all'.",
    ],
    "Several days": [
        "Okay, sounds like it happened several days. I'll note that.",
        "Thanks, I'll classify that as 'several days'.",
        "Got it — several days it is.",
    ],
    "More than half the days": [
        "I hear you, I'll put that down as 'more than half the days'.",
        "Understood — that's 'more than half the days'.",
        "Thanks for sharing, I'll classify it accordingly.",
    ],
    "Nearly every day": [
        "That sounds really tough, I’ll mark that as 'nearly every day'.",
        "Thanks for your honesty, I’ll put that down as 'nearly every day'.",
        "I appreciate you sharing that — 'nearly every day' noted.",
    ],
}

# Start interaction
print("💬 Hello, friend. I'm here to support you. Please answer the next few questions openly.\n")

user_answers = []
total_score = 0

for i, question in enumerate(phq9_questions):
    print(f"Q{i+1}: {question}")
    user_input = input("You: ")

    # Classify user response
    classification = classification_chain.invoke({"response": user_input}).strip()
    score = score_mapping.get(classification, 0)
    total_score += score

    # Empathetic reply
    empathetic_msg = empathetic_reply_chain.invoke({
        "question": question,
        "response": user_input
    })

    # Friendly conversational confirmation
    confirmation_line = random.choice(confirmation_phrases.get(classification, [f"I'll note that as {classification.lower()}"]))

    user_answers.append((question, user_input, classification))

    print(f"💬 {confirmation_line}")
    print(f"💛 {empathetic_msg}\n")

# Final summary message
formatted = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in user_answers])
final_message = final_chain.invoke({
    "all_responses": formatted,
    "score": total_score
})

print("="*60)
print("🌸 Final Supportive Message:")
print(final_message)
print("="*60)
