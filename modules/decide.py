import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def gemini_request(messages):
    try:
        response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            json={"contents": messages},
            timeout=20
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_document_answers(qa_pairs):
    """
    Uses Gemini 1.5 Flash to generate grounded answers for questions based on related clauses.
    """
    total_questions = len(qa_pairs)
    print(f"⚡ Gemini: Processing {total_questions} questions...")

    # Build prompt
    prompt_blocks = []
    for i, item in enumerate(qa_pairs, 1):
        question = item["question"]
        clauses = item.get("related_clauses", [])
        context = "\n".join(clauses[:3]) if clauses else "No relevant clauses available."
        prompt_blocks.append(f"{i}. Question: {question}\nContext:\n{context}")

    final_prompt = "\n\n".join(prompt_blocks)

    instructions = (
        "You are a legal assistant. For each question below, use only the provided clause context to answer.\n"
        "Do NOT use prior knowledge or make assumptions. If not answerable from context, reply: 'Not mentioned in context.'\n"
        "Give your answers in this exact format:\n1. <answer>\n2. <answer>\n...\n"
    )

    # Gemini message format
    messages = [
        {"role": "user", "parts": [{"text": instructions + "\n\n" + final_prompt}]}
    ]

    result = gemini_request(messages)

    if "error" in result or "candidates" not in result:
        print(result["error"])
        return {"answers": ["Answer not available."] * total_questions}

    content = result["candidates"][0]["content"]["parts"][0]["text"]
    return {"answers": turbo_parse_response(content, total_questions)}

def turbo_parse_response(content, expected_count):
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    answers = []

    for line in lines:
        if line[0].isdigit() and '.' in line:
            parts = line.split('.', 1)
            if len(parts) == 2:
                answer = parts[1].strip()
                if answer:
                    answers.append(answer[:300])  # Slightly higher cap for clarity

    while len(answers) < expected_count:
        answers.append("Answer not available.")
    return answers[:expected_count]
