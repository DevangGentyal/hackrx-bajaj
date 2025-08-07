from pinecone import Pinecone
from dotenv import load_dotenv
import os
from typing import List, Dict
import json

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Load index host and name
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")  # example: 'my-index-xxxxxx.svc.us-east1-gcp.pinecone.io'
INDEX_NAMESPACE = "default"

# Connect to index
index = pc.Index(host=INDEX_HOST)
# print(index.describe_index_stats())


def search_with_text(questions: List[str], top_k: int = 3) -> List[Dict]:
    """
    Perform semantic search and return a list of dicts:
    {
        "question": <question>,
        "related_clauses": [<chunk1>, <chunk2>, ...]
    }
    """
    final_output = []

    for question in questions:
        # print(f"\n🔍 Querying: {question}")

        result = index.search(
            namespace=INDEX_NAMESPACE,
            query={
                "inputs": {"text": question},
                "top_k": top_k
            }
        )

        hits = result.get("result", {}).get("hits", [])

        related_clauses = []
        for i, hit in enumerate(hits):
            score = hit.get("_score")
            chunk = hit.get("fields", {}).get("text", "<no text>")
            related_clauses.append(chunk)
            # print(f"  #{i+1} (score={score:.3f}):\n    {chunk[:300]}\n")

        final_output.append({
            "question": question,
            "related_clauses": related_clauses
        })

    return final_output


# Example usage
if __name__ == "__main__":
    questions = [
        "Are mental illnesses covered by the policy?",
        "What is the waiting period for pre-existing diseases?"
    ]

    output = search_with_text(questions)
    # print("\n🧾 Final JSON Output:")
    # print(json.dumps(output, indent=2))
