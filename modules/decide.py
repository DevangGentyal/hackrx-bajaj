import os
import requests
from dotenv import load_dotenv
import json
import time
import random

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
9. Never use special characters, formatting, bold, italics, asterisks, or styling - plain text only.
10. Keep the answer concise but complete.

### FORMAT:
One concise line that clearly follows:  
Decision + Explanation + Clause Proof

"""

def make_groq_request(payload, max_retries=3, base_delay=1):
    """
    Make a request to Groq API with retry logic and rate limiting
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    for attempt in range(max_retries):
        try:
            # Add small random delay to avoid hitting rate limits
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"   Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            
            response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
            
            # Handle different HTTP errors
            if response.status_code == 429:  # Rate limit
                print(f"   Rate limited. Waiting...")
                time.sleep(5)
                continue
            elif response.status_code == 401:  # Auth error
                return {"error": "API key authentication failed"}
            elif response.status_code >= 500:  # Server error
                print(f"   Server error {response.status_code}. Retrying...")
                continue
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            print(f"   Request timeout (attempt {attempt + 1})")
            continue
        except requests.exceptions.ConnectionError:
            print(f"   Connection error (attempt {attempt + 1})")
            continue
        except requests.exceptions.RequestException as e:
            print(f"   Request error (attempt {attempt + 1}): {str(e)}")
            continue
        except Exception as e:
            print(f"   Unexpected error (attempt {attempt + 1}): {str(e)}")
            continue
    
    return {"error": f"Failed after {max_retries} attempts"}


def get_document_answers(qa_pairs):
    """
    Process questions and related clauses to generate precise answers
    
    Args:
        qa_pairs (list): List of dictionaries with 'question' and 'related_clauses' keys
    
    Returns:
        dict: Dictionary with 'answers' key containing list of answer strings
    """
    answers = []
    total_questions = len(qa_pairs)
    
    print(f"Processing {total_questions} questions...")

    for i, item in enumerate(qa_pairs, 1):
        question = item["question"]
        clauses = item["related_clauses"]
        
        print(f"\n[{i}/{total_questions}] Processing: {question[:80]}...")

        # Check if all clauses are empty or placeholder
        if not clauses or all(clause.strip() in ["", "<no text>"] for clause in clauses):
            print("   No relevant clauses found")
            answers.append("Answer could not be generated due to lack of relevant document information.")
            continue

        # Build prompt - more concise to avoid token limits
        clauses_text = "\n".join(f"- {clause[:500]}..." if len(clause) > 500 else f"- {clause}" 
                                 for clause in clauses[:3])  # Limit to top 3 clauses
        
        user_prompt = f"""
QUESTION: {question}

RELEVANT CLAUSES:
{clauses_text}

Generate a single sentence answer that includes the decision, explanation, and clause proof. Use plain text only - no formatting, bold, or special characters.
"""

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Lower temperature for more consistent answers
            "max_tokens": 200,   # Shorter responses
            "top_p": 0.9
        }

        # Add small delay between requests to avoid rate limiting
        if i > 1:
            time.sleep(0.5)

        result = make_groq_request(payload)

        if "error" in result:
            print(f"   Failed: {result['error']}")
            answers.append(f"Answer could not be generated: {result['error']}")
            continue

        if "choices" not in result or not result["choices"]:
            print("   No valid response")
            answers.append("Answer could not be generated due to API response issues.")
            continue

        content = result["choices"][0]["message"]["content"].strip()

        # Clean up response
        content = content.replace('"', '').strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["answer:", "response:", "decision:", "the answer is:"]
        for prefix in prefixes_to_remove:
            if content.lower().startswith(prefix):
                content = content[len(prefix):].strip()
        
        # Remove formatting characters
        formatting_chars = ["**", "*", "_", "##", "#"]
        for char in formatting_chars:
            content = content.replace(char, "")

        if not content:
            content = "Answer could not be generated due to insufficient document context."

        print(f"   Success: {content[:100]}...")
        answers.append(content)

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

    print("🔄 Processing questions...")
    result = get_document_answers(example_data)

    print("\n📋 Final Output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print(f"\n✅ Successfully processed {len(result['answers'])} questions")