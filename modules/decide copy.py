import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """
You are a decision assistant that answers questions using only the provided clauses. Your job is to generate a **one-line** answer in the format:  

**Decision + Explanation + Clause reference as Proof**

### RULES:
1. Use only the provided clauses.
2. Return a **single sentence** per answer.
3. Answer must include: decision, explanation, and exact clause proof.
4. Always include all important figures, dates, durations, and conditions.
5. Use exact terms from clauses (do not generalize).
6. Do not add assumptions, background, or filler phrases.
7. Always include exceptions or special conditions if mentioned.
8. Never use vague terms like "as per clause 2" — instead, quote the clause title or reference exactly as given.
9. Never use special characters unless needed. Also no styling for content as bold, *, italics or something

### FORMAT:
One concise line that clearly follows:  
Decision + Explanation + Clause Proof

"""

def get_document_answers(qa_pairs):
    """
    Process questions and related clauses to generate precise answers
    
    Args:
        qa_pairs (list): List of dictionaries with 'question' and 'related_clauses' keys
    
    Returns:
        dict: Dictionary with 'answers' key containing list of answer strings
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    answers = []

    for item in qa_pairs:
        question = item["question"]
        clauses = item["related_clauses"]

        # Check if all clauses are empty or placeholder
        if not clauses or all(clause.strip() in ["", "<no text>"] for clause in clauses):
            answers.append("Answer could not be generated due to lack of relevant document information.")
            continue

        # Build prompt
        user_prompt = f"""
QUESTION: {question}

RELEVANT DOCUMENT CLAUSES:
{chr(10).join(f"• {clause}" for clause in clauses)}

TASK: Based only on the clauses above, write a 1-line answer that:
- Clearly follows this structure: Decision + Explanation + Clause Proof
- Includes all important numbers, conditions, and limitations
- Uses terms and titles exactly as written in the clauses
- Avoids any extra formatting or background

Return only the one-line answer.
"""

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500,
            "top_p": 0.9
        }

        try:
            response = requests.post(url, headers=HEADERS, json=payload)
            response.raise_for_status()
            result = response.json()

            if "choices" not in result or not result["choices"]:
                # print(f"❌ No valid response for question: {question}")
                answers.append("Answer could not be generated due to API response issues.")
                continue

            content = result["choices"][0]["message"]["content"].strip()

            # Clean up
            content = content.replace('"', '').strip()
            if content.lower().startswith("answer:"):
                content = content[7:].strip()

            if not content:
                content = "Answer could not be generated due to insufficient document context."

            answers.append(content)

        except requests.exceptions.RequestException as e:
            # print(f"❌ Request error for question '{question}': {e}")
            answers.append("Answer could not be generated due to connection error.")
        except Exception as e:
            # print(f"❌ Unexpected error for question '{question}': {e}")
            answers.append("Answer could not be generated due to an unexpected error.")

    return {
        "answers": answers
    }

# ----------------- For Testing ------------------

if __name__ == "__main__":
    example_data = [
        {
            "question": "What is the grace period for premium payment?",
            "related_clauses": [
                "A grace period of thirty (30) days is granted for the payment of renewal premium.",
                "During the grace period, the policy remains in force and all benefits continue.",
                "If premium is not paid within the grace period, the policy will lapse and coverage will cease."
            ]
        },
        {
            "question": "Are mental illnesses covered by the policy?",
            "related_clauses": [
                "Mental illness treatment is covered up to the sum insured, subject to conditions listed under section 3.5.",
                "The policy covers psychiatric consultation, admission, and medication under the mental health clause.",
                "Coverage for mental illnesses excludes certain conditions such as personality disorders and developmental delays.",
                "Pre-authorization is required for all mental health treatments exceeding Rs. 25,000."
            ]
        },
    ]

    # print("🔄 Processing questions...")
    result = get_document_answers(example_data)

    # print("\n📋 Final Output:")
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # print(f"\n✅ Successfully processed {len(result['answers'])} questions")
