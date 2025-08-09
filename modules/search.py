from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
index = pc.Index(host=INDEX_HOST)

# Load the same embedding model used during ingestion
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def lightning_search(questions: List[str], ns: str) -> List[Dict]:
    """
    Optimized semantic clause search using local embeddings for best accuracy.
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
            # Embed the query locally
            query_embedding = embed_model.encode(question).tolist()

            # Search in Pinecone using the vector
            result = index.query(
                namespace=ns,
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )

            hits = result.get("matches", [])
            clauses = []

            for hit in hits:
                text = hit.get("metadata", {}).get("text")
                # Fallback: any long-enough metadata string
                if not text:
                    text = next((v for v in hit.get("metadata", {}).values()
                                 if isinstance(v, str) and len(v) > 20), None)
                if text and text not in clauses:
                    clauses.append(text)
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
            except Exception as e:
                print(f"⚠️ Search future failed: {e}")

    # Fill missing results
    for i, result in enumerate(final_output):
        if result is None:
            final_output[i] = {"question": questions[i], "related_clauses": []}

    print("✅ Lightning search completed successfully")
    index.delete(delete_all=True, namespace=ns)
    print("🧹 Namespace cleared from index")

    return final_output

# Replace the function
search_with_text = lightning_search
