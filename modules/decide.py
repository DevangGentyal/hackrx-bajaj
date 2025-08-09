import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# Google’s Gemini usually handles ~30-32k characters safely for text
MAX_PROMPT_CHARS = 28000  

def gemini_request(messages):
    try:
        payload = {"contents": messages}
        print(f"\n📤 Sending request to Gemini API: {GEMINI_URL}")
        print(f"🔑 API key loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
        print(f"📦 Payload size: {len(json.dumps(payload))} chars")
        if len(json.dumps(payload)) > MAX_PROMPT_CHARS:
            print(f"⚠ WARNING: Payload size exceeds {MAX_PROMPT_CHARS} chars!")

        response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        print(f"📥 Status: {response.status_code}")
        print(f"📥 Raw response (truncated): {response.text[:800]}{'...' if len(response.text) > 800 else ''}")

        try:
            return response.json()
        except Exception as json_err:
            return {"error": f"Invalid JSON in Gemini response: {json_err}", "raw_response": response.text}

    except requests.Timeout:
        return {"error": "Gemini API request timed out"}
    except Exception as e:
        return {"error": f"Gemini API request failed: {e}"}


def get_document_answers(qa_pairs):
    """
    Splits qa_pairs into batches to stay within Gemini's limits.
    Returns combined answers in the correct order.
    """
    total_questions = len(qa_pairs)
    print(f"\n⚡ Gemini: Processing {total_questions} questions with batching...")

    batches = []
    current_batch = []
    current_size = 0

    # Build batches without splitting a Q&A
    for qa in qa_pairs:
        question = qa.get("question", "[No question provided]")
        clauses = qa.get("related_clauses", [])
        context = "\n".join(clauses[:3]) if clauses else "No relevant clauses available."
        block = f"Question: {question}\nContext:\n{context}\n\n"

        block_size = len(block)
        if current_size + block_size > MAX_PROMPT_CHARS and current_batch:
            batches.append(current_batch)
            current_batch = [qa]
            current_size = block_size
        else:
            current_batch.append(qa)
            current_size += block_size

    if current_batch:
        batches.append(current_batch)

    print(f"📦 Total batches: {len(batches)}")

    all_answers = []

    for b_idx, batch in enumerate(batches, start=1):
        print(f"\n🚀 Processing batch {b_idx}/{len(batches)} ({len(batch)} questions)")

        prompt_blocks = []
        for i, item in enumerate(batch, 1):
            q = item.get("question", "[No question provided]")
            clauses = item.get("related_clauses", [])
            print(f"📝 Q: {q}")
            context = "\n".join(clauses[:3]) if clauses else "No relevant clauses available."
            prompt_blocks.append(f"{i}. Question: {q}\nContext:\n{context}")

        final_prompt = "\n\n".join(prompt_blocks)
        instructions = (
            "You are a legal assistant. For each question below, use only the provided clause context to answer.\n"
            "Do NOT use prior knowledge or make assumptions. If not mentioned in Context say Docs didnt mentioned this\n"
            "Give your answers in this exact format:\n1. <answer>\n2. <answer>\n...\n"
        )

        messages = [
            {"role": "user", "parts": [{"text": instructions + "\n\n" + final_prompt}]}
        ]

        result = gemini_request(messages)

        if "error" in result or "candidates" not in result:
            print("❌ Gemini error:", result.get("error", "Unknown error"))
            all_answers.extend(["Answer not available."] * len(batch))
            continue

        try:
            content = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as parse_err:
            print("❌ Failed to parse Gemini output:", parse_err)
            all_answers.extend(["Answer not available."] * len(batch))
            continue

        answers = turbo_parse_response(content, len(batch))
        all_answers.extend(answers)

    return {"answers": all_answers}


def turbo_parse_response(content, expected_count):
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    answers = []
    for line in lines:
        if line[0].isdigit() and '.' in line:
            parts = line.split('.', 1)
            if len(parts) == 2:
                answer = parts[1].strip()
                if answer:
                    answers.append(answer[:300])

    while len(answers) < expected_count:
        answers.append("Answer not available.")

    return answers[:expected_count]
