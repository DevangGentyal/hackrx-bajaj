from pinecone import Pinecone
from dotenv import load_dotenv
import os
from typing import List, Dict
import json

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Load index host and name
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# Connect to index
index = pc.Index(host=INDEX_HOST)


def search_with_text(questions: List[str], ns: str, top_k: int = 3, max_retries: int = 5, retry_delay: int = 2) -> List[Dict]:
    """
    Perform semantic search and return a list of dicts:
    {
        "question": <question>,
        "related_clauses": [<chunk1>, <chunk2>, ...]
    }
    
    Args:
        questions: List of questions to search
        ns: Namespace to search in
        top_k: Number of results to return per question
        max_retries: Maximum number of retries if namespace not ready
        retry_delay: Delay between retries in seconds
    """
    import time
    
    final_output = []
    
    # Wait for namespace to be ready with retries
    namespace_ready = False
    for attempt in range(max_retries):
        try:
            stats = index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            
            if ns in namespaces:
                vector_count = namespaces[ns].get("vector_count", 0)
                if vector_count > 0:
                    print(f"✅ Namespace '{ns}' is ready with {vector_count} vectors")
                    namespace_ready = True
                    break
                else:
                    print(f"⏳ Namespace '{ns}' exists but empty (attempt {attempt + 1}/{max_retries})")
            else:
                print(f"⏳ Namespace '{ns}' not found yet (attempt {attempt + 1}/{max_retries})")
                print(f"   Available: {list(namespaces.keys())}")
            
            if attempt < max_retries - 1:
                print(f"   Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                
        except Exception as e:
            print(f"Error checking namespace (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    if not namespace_ready:
        error_msg = f"Namespace '{ns}' not ready after {max_retries} attempts"
        print(f"❌ {error_msg}")
        return [{"question": q, "related_clauses": [], "error": error_msg} for q in questions]
    
    print(f"----NameSpace----\n{ns}")

    for question in questions:
        print(f"\n🔍 Querying: {question}")

        # Retry search if needed (sometimes indexing takes a bit longer)
        search_successful = False
        for search_attempt in range(3):  # Max 3 search attempts
            try:
                result = index.search(
                    namespace=ns,
                    query={
                        "inputs": {"text": question},
                        "top_k": top_k
                    }
                )

                hits = result.get("result", {}).get("hits", [])
                # print(f"Found {len(hits)} hits")
                
                # If we get 0 hits, it might still be indexing
                if len(hits) == 0 and search_attempt < 2:
                    print(f"   No hits found, retrying in {retry_delay} seconds... (attempt {search_attempt + 1}/3)")
                    time.sleep(retry_delay)
                    continue
                
                search_successful = True
                
                related_clauses = []
                for i, hit in enumerate(hits):
                    score = hit.get("_score", 0)
                    fields = hit.get("fields", {})
                    
                    # Try to find text content in common field names
                    text_content = None
                    possible_fields = ["text", "chunk_text", "content", "description", "body"]
                    
                    for field_name in possible_fields:
                        if field_name in fields:
                            text_content = fields[field_name]
                            break
                    
                    # If still no text, take the first string field
                    if not text_content and fields:
                        for field_name, field_value in fields.items():
                            if isinstance(field_value, str) and len(field_value) > 10:
                                text_content = field_value
                                break
                    
                    if text_content:
                        related_clauses.append(text_content)
                        # print(f"  #{i+1} (score={score:.3f}):\n    {text_content[:300]}{'...' if len(text_content) > 300 else ''}\n")
                    else:
                        print(f"  #{i+1} (score={score:.3f}): No text found. Available fields: {list(fields.keys())}")
                        related_clauses.append(f"[No text content - ID: {hit.get('_id', 'unknown')}]")

                final_output.append({
                    "question": question,
                    "related_clauses": related_clauses
                })
                break  # Exit retry loop on success

            except Exception as e:
                print(f"Error searching for question '{question}' (attempt {search_attempt + 1}): {str(e)}")
                if search_attempt < 2:
                    time.sleep(1)
                else:
                    final_output.append({
                        "question": question,
                        "related_clauses": [],
                        "error": str(e)
                    })

    print("-------\n\nFinal Output: ",final_output)
    return final_output