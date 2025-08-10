import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "gemini-2.0-flash-lite"
RPM_LIMIT = 30          # Requests per minute
TPM_LIMIT = 1_000_000   # Tokens per minute
SAFE_PROMPT_CHARS = 28000  # ~28k to avoid 32k cutoff
SECONDS_PER_REQUEST = 60 / RPM_LIMIT

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"


# ===============================
# GEMINI REQUEST
# ===============================
def gemini_request(messages):
    payload = {"contents": messages}
    payload_str = json.dumps(payload)
    payload_size = len(payload_str)
    print(f"\n📤 Sending to Gemini ({payload_size} chars)...")
    if payload_size > SAFE_PROMPT_CHARS:
        print(f"⚠ Payload exceeds {SAFE_PROMPT_CHARS} chars! May cause issues.")

    try:
        response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        print(f"📥 Gemini Status: {response.status_code}")
        if response.status_code != 200:
            print(f"❌ Gemini API Response Text: {response.text}")
            return {"error": f"Gemini API failed with status {response.status_code}"}
        return response.json()

    except requests.Timeout:
        print("❌ Gemini API request timed out.")
        return {"error": "Gemini API request timed out"}
    except Exception as e:
        print(f"❌ Gemini API request failed: {e}")
        return {"error": f"Gemini API request failed: {e}"}


# ===============================
# DOCUMENT ANSWERS
# ===============================
def get_document_answers(qa_pairs):
    print(f"\n⚡ Gemini: Processing {len(qa_pairs)} questions...")

    batches = []
    current_batch = []
    instructions = (
        "You are an expert legal assistant. For each question below, use ONLY the provided clause context to answer.\n"
        "Do NOT use prior knowledge or make assumptions.\n"
        "If the context does not mention something, reply exactly with: Docs didn't mention this.\n"
        "Return answers in this exact format:\n1. <answer>\n2. <answer>\n...\n"
        "Each answer must follow this internal structure: Decision + Explanation + Backing (mention the exact title of the clause or section).\n"
        "IMPORTANT: You must merge these three parts into a single, flowing human-like sentence or paragraph.\n"
        "You MUST NOT include any labels, headings, or prefixes such as 'Decision:', 'Explanation:', or 'Backing:' in your final output.\n"
        "If you include any of these labels, the answer is considered WRONG.\n"
        "If the ans are too short (1-2 words) make it aleast a single sentence"
    )

    def batch_payload_size(batch):
        prompt_blocks = [
            f"{i+1}. Question: {item.get('question', '[No question provided]')}\nContext:\n" +
            ("\n".join(item.get("related_clauses", [])[:3]) if item.get("related_clauses") else "No relevant clauses available.")
            for i, item in enumerate(batch)
        ]
        final_prompt = instructions + "\n\n" + "\n\n".join(prompt_blocks)
        payload = {"contents": [{"role": "user", "parts": [{"text": final_prompt}]}]}
        return len(json.dumps(payload))

    for qa in qa_pairs:
        current_batch.append(qa)
        size = batch_payload_size(current_batch)

        if size > SAFE_PROMPT_CHARS and len(current_batch) > 1:
            # Remove last QA and finalize current batch
            removed = current_batch.pop()
            batches.append(current_batch)
            print(f"🔸 Created batch with {len(current_batch)} questions, size: {batch_payload_size(current_batch)} chars")
            current_batch = [removed]

    if current_batch:
        batches.append(current_batch)
        print(f"🔸 Created batch with {len(current_batch)} questions, size: {batch_payload_size(current_batch)} chars")

    print(f"📦 Total batches created: {len(batches)}")

    all_answers = []

    for b_idx, batch in enumerate(batches, start=1):
        try:
            print(f"\n🚀 Processing batch {b_idx}/{len(batches)} with {len(batch)} questions")

            prompt_blocks = [
                f"{i+1}. Question: {item.get('question', '[No question provided]')}\nContext:\n" +
                ("\n".join(item.get("related_clauses", [])[:3]) if item.get("related_clauses") else "No relevant clauses available.")
                for i, item in enumerate(batch)
            ]
            final_prompt = instructions + "\n\n" + "\n\n".join(prompt_blocks)

            messages = [{"role": "user", "parts": [{"text": final_prompt}]}]

            result = gemini_request(messages)

            if "error" in result:
                print(f"❌ Error from Gemini in batch {b_idx}: {result['error']}")
                all_answers.extend(["Answer not available."] * len(batch))
                continue

            if "candidates" not in result:
                print(f"❌ Gemini response missing 'candidates' in batch {b_idx}: {result}")
                all_answers.extend(["Answer not available."] * len(batch))
                continue

            try:
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"📄 Gemini raw response preview (first 200 chars):\n{content[:200]}")
            except Exception as e:
                print(f"❌ Error parsing Gemini response content: {e}")
                all_answers.extend(["Answer not available."] * len(batch))
                continue

            answers = turbo_parse_response(content, len(batch))
            all_answers.extend(answers)

        except Exception as e:
            print(f"❌ Unexpected error processing batch {b_idx}: {e}")
            all_answers.extend(["Answer not available."] * len(batch))

    return {"answers": all_answers}


# ===============================
# PARSE GEMINI RESPONSE
# ===============================
def turbo_parse_response(content, expected_count):
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    answers = []

    for line in lines:
        if len(line) > 1 and line[0].isdigit() and '.' in line:
            parts = line.split('.', 1)
            if len(parts) == 2:
                answer = parts[1].strip()
                if answer:
                    # Truncate to 300 chars, you can increase or remove if needed
                    answers.append(answer[:300])

    while len(answers) < expected_count:
        answers.append("Answer not available.")

    if len(answers) > expected_count:
        print(f"⚠ Warning: Parsed more answers ({len(answers)}) than expected ({expected_count})")

    return answers[:expected_count]


# ===============================
# TEST RUN
# ===============================
if __name__ == "__main__":
    sample_qa = [
        {
            "question": "What is the grace period for premium payment?",
            "related_clauses": ["Grace period of 30 days will be allowed..."]
        },
        {
            "question": "Does the policy cover maternity expenses?",
            "related_clauses": ["Maternity expenses are covered after a waiting period of 9 months..."]
        }
    ]
    result = get_document_answers(sample_qa)
    print(json.dumps(result, indent=2))
