from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
index = pc.Index(host=INDEX_HOST)

def lightning_search(questions: List[str], ns: str) -> List[Dict]:
    """
    Optimized semantic clause search with reliable namespace check,
    improved result filtering, and field-level clause fallback.
    """
    # Quick validation for namespace readiness
    for attempt in range(3):
        try:
            stats = index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            if ns in namespaces and namespaces[ns].get("vector_count", 0) > 0:
                print(f"⚡ Namespace ready: {namespaces[ns]['vector_count']} vectors")
                break
            elif attempt < 2:
                time.sleep(5)
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} - Stats fetch failed: {e}")
            time.sleep(5)
    else:
        print("⚠️ Proceeding without confirmed namespace...")

    print(f"🔍 Searching in namespace: {ns}\n")

    final_output = [None] * len(questions)

    def search_single_question(question_data):
        idx, question = question_data
        try:
            result = index.search(
                namespace=ns,
                query={"inputs": {"text": question}, "top_k": 5}  # Top-5 for broader match
            )
            hits = result.get("result", {}).get("hits", [])
            clauses = []

            for hit in hits:
                fields = hit.get("fields", {})
                # Fallback: get any valid string field with length > 20
                text = fields.get("text") or next((v for v in fields.values() if isinstance(v, str) and len(v) > 20), None)
                if text and text not in clauses:
                    clauses.append(text)  # Truncate to 1000 chars
                if len(clauses) >= 3:
                    break

            return idx, {"question": question, "related_clauses": clauses}
        except Exception as e:
            print(f"❌ Q{idx+1} error: {e}")
            return idx, {"question": question, "related_clauses": []}

    # Search in parallel
    with ThreadPoolExecutor(max_workers=min(10, len(questions))) as executor:
        futures = [executor.submit(search_single_question, (i, q)) for i, q in enumerate(questions)]

        for future in as_completed(futures, timeout=30):
            try:
                idx, result = future.result()
                final_output[idx] = result
                print(f"✅ Q{idx+1}: {len(result['related_clauses'])} clauses found")
                # print(result["related_clauses"])
            except Exception as e:
                print(f"⚠️ Search future failed: {e}")

    # Ensure all results are returned
    for i, result in enumerate(final_output):
        if result is None:
            final_output[i] = {"question": questions[i], "related_clauses": []}

    print("✅ Lightning search completed successfully")
    index.delete(delete_all=True, namespace=ns)
    print("🧹 Namespace cleared from index")
    # print(final_output)
    return final_output

# Replace the function
search_with_text = lightning_search