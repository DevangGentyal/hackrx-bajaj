import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

MAX_PROMPT_CHARS = 25000  # Safety limit to avoid Gemini payload errors

def gemini_request(messages):
    try:
        print(f"📤 Sending request to Gemini API ({GEMINI_URL})")
        print(f"🔑 API Key Loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
        print("📦 Request Payload:", json.dumps(messages, indent=2)[:1000] + "..." if len(json.dumps(messages)) > 1000 else json.dumps(messages, indent=2))

        response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            json={"contents": messages},
            timeout=30  # Increased timeout to 30s
        )

        print(f"📥 Response Status: {response.status_code}")
        print(f"📥 Raw Response: {response.text[:1000]}{'...' if len(response.text) > 1000 else ''}")

        # Try JSON parsing
        try:
            return response.json()
        except Exception as json_err:
            return {"error": f"Invalid JSON in Gemini response: {str(json_err)}", "raw_response": response.text}

    except requests.Timeout:
        return {"error": "Gemini API request timed out"}
    except Exception as e:
        return {"error": f"Gemini API request failed: {str(e)}"}


def get_document_answers(qa_pairs):
    """
    Uses Gemini 1.5 Flash to generate grounded answers for questions based on related clauses.
    Includes debug logging and handles large requests safely.
    """
    total_questions = len(qa_pairs)
    print(f"⚡ Gemini: Processing {total_questions} questions...")

    # Build prompt blocks
    prompt_blocks = []
    for i, item in enumerate(qa_pairs, 1):
        question = item.get("question", "[No question provided]")
        clauses = item.get("related_clauses", [])
        print(f"📝 Question {i}: {question}")
        print(f"📚 Clauses: {clauses if clauses else '[No relevant clauses]'}")
        context = "\n".join(clauses[:3]) if clauses else "No relevant clauses available."
        prompt_blocks.append(f"{i}. Question: {question}\nContext:\n{context}")

    final_prompt = "\n\n".join(prompt_blocks)

    # Trim prompt if too large for Gemini
    if len(final_prompt) > MAX_PROMPT_CHARS:
        print(f"⚠ Prompt size {len(final_prompt)} exceeds {MAX_PROMPT_CHARS} characters. Trimming...")
        final_prompt = final_prompt[:MAX_PROMPT_CHARS] + "\n\n[Content truncated due to length]"

    instructions = (
        "You are a legal assistant. For each question below, use only the provided clause context to answer.\n"
        "Do NOT use prior knowledge or make assumptions.\n"
        "Give your answers in this exact format:\n1. <answer>\n2. <answer>\n...\n"
    )

    messages = [
        {"role": "user", "parts": [{"text": instructions + "\n\n" + final_prompt}]}
    ]

    # Call Gemini
    result = gemini_request(messages)

    # Handle errors
    if not isinstance(result, dict):
        print("❌ Unexpected result type from Gemini:", type(result))
        return {"answers": ["Answer not available."] * total_questions}

    if "error" in result:
        print("❌ Gemini Error:", result["error"])
        return {"answers": ["Answer not available."] * total_questions}

    candidates = result.get("candidates")
    if not candidates or "content" not in candidates[0]:
        print("❌ Gemini returned no candidates or missing content")
        return {"answers": ["Answer not available."] * total_questions}

    try:
        content = candidates[0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as parse_err:
        print("❌ Failed to extract text from Gemini response:", str(parse_err))
        return {"answers": ["Answer not available."] * total_questions}

    return {"answers": turbo_parse_response(content, total_questions)}


def turbo_parse_response(content, expected_count):
    """
    Parses numbered list answers like:
    1. Answer A
    2. Answer B
    """
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    answers = []

    for line in lines:
        if line[0].isdigit() and '.' in line:
            parts = line.split('.', 1)
            if len(parts) == 2:
                answer = parts[1].strip()
                if answer:
                    answers.append(answer[:300])  # Limit to 300 chars for clarity

    # Pad with defaults if missing answers
    while len(answers) < expected_count:
        answers.append("Answer not available.")

    return answers[:expected_count]
