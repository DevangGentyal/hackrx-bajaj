import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def turbo_groq_request(payload):
    """Ultra-fast single request with no retries"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=HEADERS, 
            json=payload, 
            timeout=10  # Reduced timeout
        )
        return response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
    except:
        return {"error": "Request failed"}

def get_document_answers(qa_pairs, batch_size=None):
    """
    Turbo processing - all questions in single API call
    """
    total_questions = len(qa_pairs)
    print(f"⚡ Turbo processing {total_questions} questions...")

    # Ultra-compact prompt generation
    question_blocks = []
    for i, item in enumerate(qa_pairs, 1):
        question = item["question"]
        clauses = item.get("related_clauses", [])
        
        # Use only the best clause (first one) for speed
        context = clauses[0][:400] if clauses else "No context"
        question_blocks.append(f"{i}. {question}\n{context}")

    # Minimal prompt for maximum speed
    prompt = f"""Answer each question in one line. Format: "1. [answer]" etc.

{chr(10).join(question_blocks)}"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a fast AI assistant. Provide one-line answers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": total_questions * 100,  # Minimal tokens
        "top_p": 1
    }

    result = turbo_groq_request(payload)
    
    if "error" in result or "choices" not in result:
        return {"answers": ["Unable to generate answer."] * total_questions}

    # Fast response parsing
    content = result["choices"][0]["message"]["content"]
    answers = turbo_parse_response(content, total_questions)
    
    print(f"⚡ Turbo answers generated")
    return {"answers": answers}

def turbo_parse_response(content, expected_count):
    """Ultra-fast parsing"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    answers = []
    
    for line in lines:
        if line and (line[0].isdigit() or line.lower().startswith(('answer', 'q'))):
            # Extract answer after number/prefix
            if '.' in line:
                answer = line.split('.', 1)[1].strip()
            elif ':' in line:
                answer = line.split(':', 1)[1].strip()
            else:
                answer = line
            
            if answer:
                answers.append(answer[:200])  # Truncate long answers
    
    # Ensure correct count
    while len(answers) < expected_count:
        answers.append("Answer not available.")
    
    return answers[:expected_count]
