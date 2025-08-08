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
    Lightning-fast parallel search with minimal waiting
    """
    
    # Minimal namespace check - don't wait too long
    for attempt in range(3):
        try:
            stats = index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            
            if ns in namespaces and namespaces[ns].get("vector_count", 0) > 0:
                vector_count = namespaces[ns]["vector_count"]
                print(f"⚡ Namespace ready: {vector_count} vectors")
                break
            else:
                if attempt < 2:
                    time.sleep(0.5)  # Very short wait
        except:
            if attempt < 2:
                time.sleep(0.5)
    else:
        print(f"⚡ Proceeding with search anyway...")

    print(f"----NameSpace----\n{ns}\n")

    # Parallel search for all questions simultaneously
    final_output = [None] * len(questions)
    
    def search_single_question(question_data):
        idx, question = question_data
        try:
            result = index.search(
                namespace=ns,
                query={"inputs": {"text": question}, "top_k": 2}  # Reduced top_k
            )
            
            hits = result.get("result", {}).get("hits", [])
            clauses = []
            
            for hit in hits[:2]:  # Only top 2 results
                fields = hit.get("fields", {})
                text = fields.get("text") or next((v for v in fields.values() if isinstance(v, str) and len(v) > 20), None)
                if text:
                    clauses.append(text[:800])  # Truncate for speed
            
            return idx, {"question": question, "related_clauses": clauses}
        except:
            return idx, {"question": question, "related_clauses": []}

    # Process all questions in parallel
    with ThreadPoolExecutor(max_workers=min(10, len(questions))) as executor:
        futures = [
            executor.submit(search_single_question, (i, q))
            for i, q in enumerate(questions)
        ]
        
        for future in as_completed(futures, timeout=15):
            try:
                idx, result = future.result()
                final_output[idx] = result
                print(f"⚡ Q{idx+1}: Found {len(result['related_clauses'])} clauses")
            except:
                pass

    # Fill any None values
    for i, result in enumerate(final_output):
        if result is None:
            final_output[i] = {"question": questions[i], "related_clauses": []}

    print(f"⚡ Lightning search completed")
    return final_output

# Replace the function
search_with_text = lightning_search
